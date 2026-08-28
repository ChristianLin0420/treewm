"""TensorBoard wrapper.

Rank-0-only output by construction: on non-zero ranks backend operations are no-ops, so
call sites never need ``if rank == 0`` guards. Scalar identity checks remain local to
every process. ``add_hparams`` writes to a *separate* writer (``<run_dir>/hparams``) so
the final-eval hparam entry does not clutter training curves (spec section 21).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import threading
from typing import Any

import numpy as np
import torch


class ScalarCollisionError(RuntimeError):
    """One process attempted to give a scalar boundary two different values."""


class TreeWMLogger:
    def __init__(
        self,
        run_dir: str | Path,
        is_main: bool = True,
        flush_secs: int = 30,
        *,
        wandb_project: str | None = None,
        wandb_id: str | None = None,
        wandb_name: str | None = None,
        wandb_group: str | None = None,
        wandb_config: dict[str, Any] | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.is_main = is_main
        self._writer = None
        self._hparam_writer = None
        self._wandb_run = None
        self._wandb_error_reported = False
        # TensorBoard stores simple scalars as float32. Keep the latest exact
        # (step, bits) for each process-local tag. Monotonic-step enforcement makes this
        # a bounded ledger even across a formal million-update run: an equal boundary is
        # identity-checked and an older boundary is rejected rather than replayed.
        self._scalar_ledger: dict[str, tuple[int, int]] = {}
        self._scalar_ledger_lock = threading.Lock()
        if self.is_main:
            from torch.utils.tensorboard import SummaryWriter

            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=str(self.run_dir), flush_secs=flush_secs)
            if wandb_project is not None:
                try:
                    import os
                    import wandb

                    # Let W&B resolve authentication from its normal sources (including
                    # a mode-600 ~/.netrc). Only an explicit WANDB_MODE may change online
                    # behaviour; the trainer never reads or persists credentials.
                    mode = os.environ.get("WANDB_MODE")
                    self._wandb_run = wandb.init(
                        project=wandb_project,
                        entity=os.environ.get("WANDB_ENTITY") or None,
                        id=wandb_id,
                        name=wandb_name,
                        group=wandb_group,
                        resume="allow",
                        dir=str(self.run_dir),
                        config=wandb_config,
                        mode=mode,
                        # Suppress Python-SDK convenience links and minimize observational
                        # links in the scientific tree. The pinned Go core may retain its
                        # single debug-core leaf, which the reporter authenticates without
                        # following its target.
                        settings=wandb.Settings(symlink=False),
                        reinit=True,
                    )
                    self._wandb_run.define_metric("global_step")
                    self._wandb_run.define_metric("*", step_metric="global_step")
                except Exception as exc:
                    # A service-side throttle/outage must not discard an allocation's
                    # scientific updates. TensorBoard remains durable locally and the
                    # next requeue retries the same stable W&B id.
                    self._wandb_run = None
                    print(f"[treewm] W&B initialization failed; local logging continues: {exc}")

    # ------------------------------------------------------------------ scalars

    def scalar(self, tag: str, value: float, step: int) -> None:
        clean = self._new_scalars({tag: value}, step)
        if not clean:
            return
        name, value = next(iter(clean.items()))
        if self._writer is not None:
            self._writer.add_scalar(name, value, int(step))
        if self._wandb_run is not None:
            self._wandb_log({"global_step": int(step), name: value})

    def scalars(self, values: dict[str, float], step: int, prefix: str = "") -> None:
        clean = self._new_scalars(values, step, prefix=prefix)
        for name, value in clean.items():
            if self._writer is not None:
                self._writer.add_scalar(name, value, int(step))
        if self._wandb_run is not None and clean:
            self._wandb_log({"global_step": int(step), **clean})

    def _new_scalars(
        self,
        values: Mapping[str, float],
        step: int,
        *,
        prefix: str = "",
    ) -> dict[str, float]:
        """Return only new finite scalars after an atomic collision preflight.

        Values are quantized exactly as TensorBoard simple scalars are.  An identical
        repeat is observationally redundant and is suppressed for both backends; a
        different float32 bit pattern raises before either backend sees this call.
        """
        if type(step) is not int or step < 0:
            raise ValueError("scalar step must be a non-negative built-in integer")
        scalar_step = step
        candidates: dict[str, tuple[float, int]] = {}
        for tag, raw_value in values.items():
            name = f"{prefix}{tag}" if prefix else tag
            value = float(raw_value)
            if not np.isfinite(value):
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                value32 = np.float32(value)
            if not np.isfinite(value32):
                continue
            bits = int(np.asarray(value32).view(np.uint32).item())
            prior_candidate = candidates.get(name)
            if prior_candidate is not None:
                if prior_candidate[1] != bits:
                    raise ScalarCollisionError(
                        f"conflicting scalar duplicate {name}@{scalar_step} within batch: "
                        f"0x{prior_candidate[1]:08x} != 0x{bits:08x}"
                    )
                continue
            candidates[name] = (float(value32), bits)

        fresh: list[tuple[str, float, int]] = []
        with self._scalar_ledger_lock:
            # Preflight the whole batch before changing the ledger.  In particular, a
            # conflict in the last mapping entry cannot partially emit earlier entries.
            for name, (value, bits) in candidates.items():
                previous = self._scalar_ledger.get(name)
                if previous is None:
                    fresh.append((name, value, bits))
                    continue
                previous_step, previous_bits = previous
                if scalar_step < previous_step:
                    raise ScalarCollisionError(
                        f"out-of-order scalar {name}@{scalar_step}: "
                        f"latest step is {previous_step}"
                    )
                if scalar_step > previous_step:
                    fresh.append((name, value, bits))
                    continue
                if previous_bits != bits:
                    raise ScalarCollisionError(
                        f"conflicting scalar duplicate {name}@{scalar_step}: "
                        f"0x{previous_bits:08x} != 0x{bits:08x}"
                    )
            for name, _, bits in fresh:
                self._scalar_ledger[name] = (scalar_step, bits)
        return {name: value for name, value, _ in fresh}

    def _wandb_log(self, payload: dict[str, Any]) -> None:
        try:
            self._wandb_run.log(payload)
        except Exception as exc:
            # A transient tracking outage must not invalidate a scientific update or
            # prevent the durable local checkpoint from being written.
            if not self._wandb_error_reported:
                print(f"[treewm] W&B logging failed; local training continues: {exc}")
                self._wandb_error_reported = True

    # --------------------------------------------------------------- histograms

    def histogram(self, tag: str, values: torch.Tensor | np.ndarray, step: int) -> None:
        if self._writer is None:
            return
        if torch.is_tensor(values):
            values = values.detach().float().cpu().numpy()
        values = np.asarray(values).ravel()
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        self._writer.add_histogram(tag, values, step)

    def histograms(self, values: dict[str, Any], step: int) -> None:
        for tag, v in values.items():
            self.histogram(tag, v, step)

    # -------------------------------------------------------------- images/text

    def image(self, tag: str, image: np.ndarray, step: int, dataformats: str = "HWC") -> None:
        if self._writer is None:
            return
        self._writer.add_image(tag, image, step, dataformats=dataformats)

    def figure(self, tag: str, figure, step: int, close: bool = True) -> None:
        if self._writer is None:
            if close:
                import matplotlib.pyplot as plt

                plt.close(figure)
            return
        self._writer.add_figure(tag, figure, step, close=close)

    def text(self, tag: str, text: str, step: int = 0) -> None:
        if self._writer is None:
            return
        # TensorBoard renders markdown; indent so YAML keeps its formatting.
        self._writer.add_text(tag, f"<pre>{text}</pre>", step)

    # ------------------------------------------------------------------ hparams

    def hparams(self, hparam_dict: dict[str, Any], metric_dict: dict[str, float]) -> None:
        """Written to a separate writer so training curves stay clean."""
        if not self.is_main:
            return
        from torch.utils.tensorboard import SummaryWriter

        if self._hparam_writer is None:
            self._hparam_writer = SummaryWriter(log_dir=str(self.run_dir / "hparams"))
        clean_h = {k: _hparam_value(v) for k, v in hparam_dict.items()}
        clean_m = {k: float(v) for k, v in metric_dict.items() if np.isfinite(float(v))}
        self._hparam_writer.add_hparams(clean_h, clean_m)
        self._hparam_writer.flush()

    # -------------------------------------------------------------------- admin

    def flush(self) -> None:
        if self._writer is not None:
            self._writer.flush()

    def mark_preempting(self) -> None:
        if self._wandb_run is None:
            return
        import wandb

        wandb.mark_preempting()

    def close(self, exit_code: int = 0) -> None:
        if self._writer is not None:
            self._writer.close()
        if self._hparam_writer is not None:
            self._hparam_writer.close()
        if self._wandb_run is not None:
            try:
                self._wandb_run.finish(exit_code=exit_code)
            except Exception as exc:
                print(f"[treewm] W&B finish failed after local flush: {exc}")
            self._wandb_run = None

    def __enter__(self) -> "TreeWMLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _hparam_value(v: Any) -> Any:
    """TensorBoard hparams accepts only int/float/str/bool/torch.Tensor."""
    if isinstance(v, (int, float, str, bool)):
        return v
    if v is None:
        return "none"
    return str(v)
