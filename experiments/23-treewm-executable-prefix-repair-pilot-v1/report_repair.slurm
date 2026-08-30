#!/bin/bash
#SBATCH --job-name=treewm-exp23-launch8-report-repair-placeholder
#SBATCH --partition=cpu
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --open-mode=append

set -Eeuo pipefail
umask 077

if [[ "$#" -ne 7 ]]; then
    echo "report repair archive requires snapshot_root submission_root submission_sha256 attempt source_archive source_archive_sha256 source_archive_size" >&2
    exit 2
fi
SNAPSHOT_ROOT="$1"
SUBMISSION_ROOT="$2"
SUBMISSION_SHA256="$3"
ATTEMPT="$4"
SOURCE_ARCHIVE="$5"
EXPECTED_ARCHIVE_SHA256="$6"
EXPECTED_ARCHIVE_SIZE="$7"
SPOOL_ARCHIVE="${BASH_SOURCE[0]}"
PYTHON_EXECUTABLE="/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python"

[[ -n "${SLURM_JOB_ID+x}" && "$SLURM_JOB_ID" =~ ^[1-9][0-9]*$ ]] || {
    echo "repair publication requires an exact active scalar SLURM_JOB_ID" >&2
    exit 2
}
[[ -z "${SLURM_ARRAY_JOB_ID+x}" && -z "${SLURM_ARRAY_TASK_ID+x}" ]] || {
    echo "repair publication cannot run as an array task" >&2
    exit 2
}
[[ -z "${SLURM_RESTART_COUNT+x}" || "$SLURM_RESTART_COUNT" == "0" ]] || {
    echo "repair publication cannot be requeued" >&2
    exit 2
}
[[ "$SUBMISSION_SHA256" =~ ^[0-9a-f]{64}$ && "$ATTEMPT" == "2" ]] || {
    echo "repair publication identity is malformed" >&2
    exit 2
}
[[ "$SOURCE_ARCHIVE" == "$SUBMISSION_ROOT/REPORT_REPAIR_0002_SOURCE_ARCHIVE.bin" ]] || {
    echo "sealed repair source archive path differs" >&2
    exit 2
}
[[ "$EXPECTED_ARCHIVE_SHA256" =~ ^[0-9a-f]{64}$ && "$EXPECTED_ARCHIVE_SIZE" =~ ^[1-9][0-9]*$ ]] || {
    echo "sealed repair source archive inventory is malformed" >&2
    exit 2
}
[[ -d "$SNAPSHOT_ROOT" && ! -L "$SNAPSHOT_ROOT" && -d "$SUBMISSION_ROOT" && ! -L "$SUBMISSION_ROOT" ]] || {
    echo "repair snapshot/submission root is unavailable or symlinked" >&2
    exit 2
}
[[ -f "$SPOOL_ARCHIVE" && ! -L "$SPOOL_ARCHIVE" && -x "$PYTHON_EXECUTABLE" ]] || {
    echo "spooled repair source archive/Python is unavailable" >&2
    exit 2
}

AUTHORIZATION="$SUBMISSION_ROOT/REPORT_REPAIR_0002_AUTHORIZED.json"
[[ -f "$AUTHORIZATION" && ! -L "$AUTHORIZATION" ]] || {
    echo "repair authorization is unavailable" >&2
    exit 2
}
AUTHORIZATION_SHA256="$(/usr/bin/sha256sum -- "$AUTHORIZATION")"
AUTHORIZATION_SHA256="${AUTHORIZATION_SHA256%% *}"
[[ "$AUTHORIZATION_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
    echo "repair authorization hash is malformed" >&2
    exit 2
}

export PATH="/cm/shared/apps/slurm/current/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
cd "$SNAPSHOT_ROOT"
exec "$PYTHON_EXECUTABLE" -I -S -B -c '
import base64, hashlib, json, os, stat, sys

archive_path, expected_sha, expected_size_text, canonical_archive_path = sys.argv[1:5]
expected_size = int(expected_size_text)
if not 0 < expected_size <= 8 * 1024 * 1024:
    raise SystemExit("sealed repair source archive size is out of bounds")
marker = b"__TREEWM_EXP23_SOURCE_ARCHIVE_V2_PAYLOAD__\n"
end = b"\n__TREEWM_EXP23_SOURCE_ARCHIVE_V2_END__\n"
archive_fd = os.open(
    archive_path,
    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
)
before = os.fstat(archive_fd)
payload = bytearray()
while len(payload) <= expected_size:
    chunk = os.read(archive_fd, min(1024 * 1024, expected_size + 1 - len(payload)))
    if not chunk:
        break
    payload.extend(chunk)
after = os.fstat(archive_fd)
raw = bytes(payload)
if not (
    stat.S_ISREG(before.st_mode)
    and before.st_size == expected_size
    and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    and len(raw) == expected_size
    and hashlib.sha256(raw).hexdigest() == expected_sha
    and raw.count(marker) == 1
    and raw.endswith(end)
):
    raise SystemExit("sealed repair source archive identity/hash differs")
prefix, encoded_tail = raw.split(marker, 1)
prefix += marker
body = encoded_tail[:-len(end)]
try:
    value = json.loads(body.decode("ascii"))
except Exception as exc:
    raise SystemExit(f"sealed repair source archive JSON differs: {exc}")
canonical = json.dumps(
    value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
).encode("ascii")
if not (
    canonical == body
    and isinstance(value, dict)
    and set(value) == {"archive_kind", "schema_version", "authority", "files"}
    and value.get("archive_kind") == "treewm_exp23_report_repair_source"
    and type(value.get("schema_version")) is int
    and value.get("schema_version") == 2
    and isinstance(value.get("authority"), dict)
    and isinstance(value.get("files"), dict)
    and set(value["files"])
        == {"protocol.sha256", "report.py", "report_repair.py", "report_repair.slurm"}
):
    raise SystemExit("sealed repair source archive envelope differs")
decoded = {}
projection = {}
for name in sorted(value["files"]):
    row = value["files"][name]
    if not (
        isinstance(row, dict)
        and set(row) == {"data_base64", "mode", "sha256", "size"}
        and row.get("mode") == 0o444
        and type(row.get("size")) is int
        and row["size"] >= 0
        and isinstance(row.get("sha256"), str)
        and isinstance(row.get("data_base64"), str)
    ):
        raise SystemExit(f"sealed repair source row differs: {name}")
    try:
        item = base64.b64decode(row["data_base64"], validate=True)
    except Exception as exc:
        raise SystemExit(f"sealed repair source base64 differs: {name}: {exc}")
    if len(item) != row["size"] or hashlib.sha256(item).hexdigest() != row["sha256"]:
        raise SystemExit(f"sealed repair source bytes differ: {name}")
    decoded[name] = item
    projection[name] = {"mode": 0o444, "sha256": row["sha256"], "size": row["size"]}
authority = value["authority"]
authority_projection = json.dumps(
    projection, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
).encode("ascii")
if not (
    set(authority)
    == {
        "repair_package_protocol_sha256",
        "repair_source_commit",
        "repair_source_files",
        "repair_source_files_sha256",
        "repair_source_installation_method",
        "schema_version",
    }
    and authority.get("schema_version") == 2
    and authority.get("repair_source_installation_method")
        == "direct_final_source_archive_o_excl"
    and authority.get("repair_source_files") == projection
    and authority.get("repair_source_files_sha256")
        == hashlib.sha256(authority_projection).hexdigest()
    and decoded["report_repair.slurm"] == prefix
    and decoded["protocol.sha256"]
        == (authority["repair_package_protocol_sha256"] + "\n").encode("ascii")
):
    raise SystemExit("sealed repair source authority differs")
report_bytes = decoded["report.py"]
synthetic_file = canonical_archive_path + "::report.py"
sys.argv = [synthetic_file, *sys.argv[5:]]
namespace = {
    "__name__": "__main__",
    "__file__": synthetic_file,
    "__package__": None,
    "__EXP23_REPAIR_SOURCE_ARCHIVE_FD__": archive_fd,
    "__EXP23_REPAIR_SOURCE_ARCHIVE_PATH__": canonical_archive_path,
    "__EXP23_REPAIR_SOURCE_ARCHIVE_EXECUTION_PATH__": archive_path,
    "__EXP23_REPAIR_SOURCE_ARCHIVE_SHA256__": expected_sha,
    "__EXP23_REPAIR_SOURCE_ARCHIVE_SIZE__": expected_size,
    "__EXP23_REPAIR_SOURCE_ARCHIVE_VALUE__": value,
}
exec(compile(report_bytes, synthetic_file, "exec"), namespace, namespace)
' "$SPOOL_ARCHIVE" "$EXPECTED_ARCHIVE_SHA256" "$EXPECTED_ARCHIVE_SIZE" "$SOURCE_ARCHIVE" \
    --publish-repair \
    --snapshot-root "$SNAPSHOT_ROOT" \
    --submission-root "$SUBMISSION_ROOT" \
    --submission-sha256 "$SUBMISSION_SHA256" \
    --repair-attempt "$ATTEMPT" \
    --repair-authorization-sha256 "$AUTHORIZATION_SHA256"
exit 2
: <<'__TREEWM_EXP23_SOURCE_ARCHIVE_V2_END__'
__TREEWM_EXP23_SOURCE_ARCHIVE_V2_PAYLOAD__
