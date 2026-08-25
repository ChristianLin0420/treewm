"""TensorBoard wrapper.

Rank-0-only by construction: on non-zero ranks every method is a no-op, so call sites
never need ``if rank == 0`` guards. ``add_hparams`` writes to a *separate* writer
(``<run_dir>/hparams``) so the final-eval hparam entry does not clutter training curves
(spec section 21).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


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
        value = float(value)
        if not np.isfinite(value):
            return
        if self._writer is not None:
            self._writer.add_scalar(tag, value, step)
        if self._wandb_run is not None:
            self._wandb_log({"global_step": int(step), tag: value})

    def scalars(self, values: dict[str, float], step: int, prefix: str = "") -> None:
        clean: dict[str, float] = {}
        for tag, value in values.items():
            name = f"{prefix}{tag}" if prefix else tag
            value = float(value)
            if not np.isfinite(value):
                continue
            clean[name] = value
            if self._writer is not None:
                self._writer.add_scalar(name, value, step)
        if self._wandb_run is not None and clean:
            self._wandb_log({"global_step": int(step), **clean})

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
