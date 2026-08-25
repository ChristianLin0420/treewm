#!/usr/bin/env python3
"""Race-free OGBench data preflight and resumable downloader.

The formal Slurm job uses ``--check`` only.  Run ``--download`` ahead of time
on a login/data-transfer node.  Downloads use per-dataset advisory locks,
``.part`` files, HTTP Range resume, fsync, and atomic rename, so multiple
campaign submissions cannot expose a half-written NPZ to trainers.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import http.client
import os
import signal
import socket
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from campaign import large_dataset_specs, load_manifest, standard_dataset_names


CHUNK_BYTES = 8 * 1024 * 1024
REQUIRED_NPZ_MEMBERS = {"observations.npy", "actions.npy", "terminals.npy"}
STOP_REQUESTED = False


class GracefulDataStop(RuntimeError):
    pass


class RetryableDownload(RuntimeError):
    pass


class RetryableDownloadExhausted(RuntimeError):
    """Transient source failures exhausted this allocation's retry budget."""


def _request_stop(signum: int, frame: object) -> None:
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True


def is_valid_npz(path: Path) -> bool:
    """Cheap structural validation without loading multi-GB arrays."""

    try:
        if path.stat().st_size <= 0 or not zipfile.is_zipfile(path):
            return False
        with zipfile.ZipFile(path, "r") as archive:
            return REQUIRED_NPZ_MEMBERS.issubset(set(archive.namelist()))
    except (FileNotFoundError, OSError, zipfile.BadZipFile):
        return False


def standard_paths(manifest: Mapping, data_root: Path) -> list[Path]:
    directory = data_root / manifest["data"]["standard_subdir"]
    paths: list[Path] = []
    for dataset_name in standard_dataset_names(manifest):
        stem = dataset_name.removesuffix(".npz")
        paths.extend((directory / f"{stem}.npz", directory / f"{stem}-val.npz"))
    return paths


def large_paths(manifest: Mapping, data_root: Path) -> list[Path]:
    root = data_root / manifest["data"]["large_subdir"]
    paths: list[Path] = []
    for spec in large_dataset_specs(manifest):
        directory = root / spec["directory"]
        for shard in range(spec["expected_train_shards"]):
            paths.append(directory / f"{spec['file_stem']}-{shard:03d}.npz")
            paths.append(directory / f"{spec['file_stem']}-{shard:03d}-val.npz")
    return paths


def expected_paths(manifest: Mapping, data_root: Path) -> list[Path]:
    return standard_paths(manifest, data_root) + large_paths(manifest, data_root)


def check_all_data(manifest: Mapping, data_root: Path | str) -> list[Path]:
    root = Path(data_root).resolve()
    return [path for path in expected_paths(manifest, root) if not is_valid_npz(path)]


@contextlib.contextmanager
def dataset_lock(path: Path, timeout_seconds: int) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    with path.open("a+", encoding="utf-8") as handle:
        while True:
            if STOP_REQUESTED:
                raise GracefulDataStop("scheduler requested stop while waiting for dataset lock")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for data lock {path}")
                time.sleep(1)
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} host={socket.gethostname()} acquired={time.time()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _open_resume(url: str, offset: int, timeout_seconds: int):
    headers = {"User-Agent": "treewm-rql-data-preflight/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(request, timeout=timeout_seconds)


def _download_file_once(url: str, destination: Path, *, timeout_seconds: int) -> None:
    if STOP_REQUESTED:
        raise GracefulDataStop("scheduler requested resumable data-stage stop")
    if is_valid_npz(destination):
        return
    if destination.exists():
        raise RuntimeError(
            f"invalid final file already exists: {destination}; move it aside explicitly before retrying"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    print(f"download {url} -> {destination} (resume offset {offset})", flush=True)

    try:
        response = _open_resume(url, offset, timeout_seconds)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and is_valid_npz(partial):
            os.replace(partial, destination)
            return
        if exc.code == 416 and partial.exists():
            # The partial is at/past the remote length but lacks a valid ZIP
            # central directory, so no Range request can repair it. Restart
            # only this explicitly temporary file on the next bounded retry.
            with partial.open("wb"):
                pass
            raise RetryableDownload(f"range exhausted for invalid partial {partial}; restarting") from exc
        if exc.code in {408, 425, 429, 500, 502, 503, 504}:
            raise RetryableDownload(f"transient HTTP {exc.code} for {url}") from exc
        raise

    with response:
        status = getattr(response, "status", response.getcode())
        # Some servers ignore Range.  Restart the .part file rather than append
        # a second full response and corrupt it.
        append = bool(offset and status == 206)
        mode = "ab" if append else "wb"
        with partial.open(mode) as handle:
            try:
                while True:
                    if STOP_REQUESTED:
                        raise GracefulDataStop("scheduler requested resumable data-stage stop")
                    try:
                        chunk = response.read(CHUNK_BYTES)
                    except http.client.IncompleteRead as exc:
                        # IncompleteRead owns bytes that were received but not
                        # returned by read().  Make them durable before the
                        # bounded Range retry continues from the new offset.
                        if exc.partial:
                            handle.write(exc.partial)
                        raise
                    if not chunk:
                        break
                    handle.write(chunk)
            finally:
                # Preserve every completed chunk even after a socket timeout,
                # connection reset, or scheduler stop.
                handle.flush()
                os.fsync(handle.fileno())
    if not is_valid_npz(partial):
        raise RetryableDownload(f"response ended before a complete NPZ was received: {partial}")
    os.replace(partial, destination)


def download_file(
    url: str,
    destination: Path,
    *,
    timeout_seconds: int,
    retries: int = 20,
) -> None:
    if retries < 1:
        raise ValueError("retries must be positive")
    for attempt in range(1, retries + 1):
        if STOP_REQUESTED:
            raise GracefulDataStop("scheduler requested resumable data-stage stop")
        try:
            _download_file_once(url, destination, timeout_seconds=timeout_seconds)
            return
        except GracefulDataStop:
            raise
        except urllib.error.HTTPError:
            # _download_file_once converts only transient HTTP statuses; a
            # missing/forbidden shard is a manifest/server error, not a retry.
            raise
        except (
            RetryableDownload,
            urllib.error.URLError,
            TimeoutError,
            socket.timeout,
            ConnectionError,
            http.client.HTTPException,
        ) as exc:
            if STOP_REQUESTED:
                raise GracefulDataStop("scheduler requested resumable data-stage stop") from exc
            if attempt == retries:
                raise RetryableDownloadExhausted(
                    f"download paused after {retries} transient attempts: {url}: {exc}"
                ) from exc
            delay = min(60, 2 ** min(attempt - 1, 6))
            partial = destination.with_name(destination.name + ".part")
            offset = partial.stat().st_size if partial.exists() else 0
            print(
                f"transient download failure ({attempt}/{retries}) at offset {offset}: {exc}; "
                f"retrying in {delay}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)


def download_standard(
    manifest: Mapping,
    data_root: Path,
    timeout_seconds: int,
    lock_timeout: int,
    retries: int,
) -> None:
    base_url = manifest["data"]["base_url"].rstrip("/")
    paths = standard_paths(manifest, data_root)
    lock_path = data_root / ".locks" / "standard.lock"
    with dataset_lock(lock_path, lock_timeout):
        for path in paths:
            download_file(f"{base_url}/{path.name}", path, timeout_seconds=timeout_seconds, retries=retries)


def download_large(
    manifest: Mapping,
    data_root: Path,
    timeout_seconds: int,
    lock_timeout: int,
    retries: int,
) -> None:
    base_url = manifest["data"]["base_url"].rstrip("/")
    large_root = data_root / manifest["data"]["large_subdir"]
    for spec in large_dataset_specs(manifest):
        directory = large_root / spec["directory"]
        lock_path = data_root / ".locks" / f"{spec['directory']}.lock"
        with dataset_lock(lock_path, lock_timeout):
            for shard in range(spec["expected_train_shards"]):
                names = (
                    f"{spec['file_stem']}-{shard:03d}.npz",
                    f"{spec['file_stem']}-{shard:03d}-val.npz",
                )
                for name in names:
                    url = f"{base_url}/{spec['directory']}/{name}"
                    download_file(url, directory / name, timeout_seconds=timeout_seconds, retries=retries)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--download", action="store_true")
    parser.add_argument("--manifest", type=Path, default=here / "manifest.json")
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("RQL_DATA_ROOT", here / "data")))
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--retries", type=int, default=20)
    parser.add_argument("--lock-timeout-seconds", type=int, default=3600)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    signal.signal(signal.SIGUSR1, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    try:
        manifest = load_manifest(args.manifest)
        data_root = args.data_root.resolve()
        if args.download:
            download_standard(
                manifest,
                data_root,
                args.timeout_seconds,
                args.lock_timeout_seconds,
                args.retries,
            )
            download_large(
                manifest,
                data_root,
                args.timeout_seconds,
                args.lock_timeout_seconds,
                args.retries,
            )
        missing = check_all_data(manifest, data_root)
        if missing:
            print(f"data preflight failed: {len(missing)} missing or invalid files", file=sys.stderr)
            for path in missing[:20]:
                print(f"  {path}", file=sys.stderr)
            if len(missing) > 20:
                print(f"  ... and {len(missing) - 20} more", file=sys.stderr)
            return 2
        print(f"data preflight OK: {len(expected_paths(manifest, data_root))} NPZ files")
        return 0
    except GracefulDataStop as exc:
        print(f"data preparation paused safely: {exc}", file=sys.stderr)
        return 75
    except RetryableDownloadExhausted as exc:
        # The stage batch wrapper explicitly requeues exit 75.  This preserves
        # all fsynced Range progress across a prolonged upstream outage while
        # still allowing fatal HTTP errors (for example 403/404) to stop.
        print(f"data preparation paused safely: {exc}", file=sys.stderr)
        return 75
    except (OSError, RuntimeError, TimeoutError, urllib.error.URLError) as exc:
        print(f"data preparation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
