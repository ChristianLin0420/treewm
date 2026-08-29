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

if [[ "$#" -ne 8 ]]; then
    echo "report_repair.slurm requires snapshot_root submission_root submission_sha256 attempt source_root report_program report_sha256 report_size" >&2
    exit 2
fi
SNAPSHOT_ROOT="$1"
SUBMISSION_ROOT="$2"
SUBMISSION_SHA256="$3"
ATTEMPT="$4"
SOURCE_ROOT="$5"
REPORT_PROGRAM="$6"
EXPECTED_REPORT_SHA256="$7"
EXPECTED_REPORT_SIZE="$8"
PYTHON_EXECUTABLE="/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python"

[[ -n "${SLURM_JOB_ID+x}" && "$SLURM_JOB_ID" =~ ^[1-9][0-9]*$ ]] || {
    echo "repair publication requires an exact active scalar SLURM_JOB_ID" >&2
    exit 2
}
[[ -z "${SLURM_ARRAY_JOB_ID+x}" && -z "${SLURM_ARRAY_TASK_ID+x}" ]] || {
    echo "repair publication cannot run as an array task" >&2
    exit 2
}
[[ -n "${SLURM_RESTART_COUNT+x}" && "$SLURM_RESTART_COUNT" == "0" ]] || {
    echo "repair publication cannot be requeued" >&2
    exit 2
}
[[ "$SUBMISSION_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
    echo "repair publication submission hash is malformed" >&2
    exit 2
}
[[ "$ATTEMPT" == "1" ]] || {
    echo "only report repair attempt 1 is authorized" >&2
    exit 2
}
[[ "$SOURCE_ROOT" == "$SUBMISSION_ROOT/report-repair/attempt-0001/source" && "$REPORT_PROGRAM" == "$SOURCE_ROOT/report.py" ]] || {
    echo "sealed repair publisher path differs" >&2
    exit 2
}
[[ "$EXPECTED_REPORT_SHA256" =~ ^[0-9a-f]{64}$ && "$EXPECTED_REPORT_SIZE" =~ ^[1-9][0-9]*$ ]] || {
    echo "sealed repair publisher inventory is malformed" >&2
    exit 2
}
[[ -d "$SNAPSHOT_ROOT" && ! -L "$SNAPSHOT_ROOT" && -d "$SUBMISSION_ROOT" && ! -L "$SUBMISSION_ROOT" ]] || {
    echo "repair snapshot/submission root is unavailable or symlinked" >&2
    exit 2
}
[[ -f "$REPORT_PROGRAM" && ! -L "$REPORT_PROGRAM" && -x "$PYTHON_EXECUTABLE" ]] || {
    echo "sealed repair publisher/Python is unavailable" >&2
    exit 2
}

# The controller seals RELEASED only after the held job is authorized and the
# release call returns.  report.py performs the bounded monotonic wait because
# the batch process can start in the narrow interval between those events.  Its
# three-hour evidence budget is bound by the authorization and leaves one hour
# of this four-hour allocation for deterministic assembly and publication.
AUTHORIZATION="$SUBMISSION_ROOT/journal/REPORT_REPAIR_0001_AUTHORIZED.json"
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
# The inline bootstrap opens report.py relative to a no-follow directory FD,
# hashes and reads that same FD, and executes the verified bytes.  This avoids
# importing through a path that could change after the controller sealed its
# source inventory and submitted the spool copy of this batch script.
exec "$PYTHON_EXECUTABLE" -I -S -B -c '
import hashlib, os, stat, sys
report_path, expected_sha, expected_size_text = sys.argv[1:4]
expected_size = int(expected_size_text)
source_root = os.path.dirname(report_path)
root_fd = os.open(
    source_root,
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
)
try:
    root_info = os.fstat(root_fd)
    if not (
        stat.S_ISDIR(root_info.st_mode)
        and stat.S_IMODE(root_info.st_mode) == 0o555
        and root_info.st_uid == os.getuid()
    ):
        raise SystemExit("sealed repair source root identity differs")
    report_fd = os.open(
        "report.py",
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=root_fd,
    )
    try:
        before = os.fstat(report_fd)
        payload = bytearray()
        while len(payload) <= expected_size:
            chunk = os.read(report_fd, min(1024 * 1024, expected_size + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        after = os.fstat(report_fd)
    finally:
        os.close(report_fd)
finally:
    os.close(root_fd)
if not (
    stat.S_ISREG(before.st_mode)
    and stat.S_IMODE(before.st_mode) == 0o444
    and before.st_uid == os.getuid()
    and before.st_nlink == 1
    and before.st_size == expected_size
    and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    and len(payload) == expected_size
    and hashlib.sha256(payload).hexdigest() == expected_sha
):
    raise SystemExit("sealed repair publisher identity/hash differs")
sys.argv = [report_path, *sys.argv[4:]]
namespace = {"__name__": "__main__", "__file__": report_path, "__package__": None}
exec(compile(bytes(payload), report_path, "exec"), namespace, namespace)
' "$REPORT_PROGRAM" "$EXPECTED_REPORT_SHA256" "$EXPECTED_REPORT_SIZE" \
    --publish-repair \
    --snapshot-root "$SNAPSHOT_ROOT" \
    --submission-root "$SUBMISSION_ROOT" \
    --submission-sha256 "$SUBMISSION_SHA256" \
    --repair-attempt "$ATTEMPT" \
    --repair-authorization-sha256 "$AUTHORIZATION_SHA256"
