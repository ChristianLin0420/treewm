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
    def __init__(self, run_dir: str | Path, is_main: bool = True, flush_secs: int = 30) -> None:
        self.run_dir = Path(run_dir)
        self.is_main = is_main
        self._writer = None
        self._hparam_writer = None
        if self.is_main:
            from torch.utils.tensorboard import SummaryWriter

            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=str(self.run_dir), flush_secs=flush_secs)

    # ------------------------------------------------------------------ scalars

    def scalar(self, tag: str, value: float, step: int) -> None:
        if self._writer is None:
            return
        value = float(value)
        if not np.isfinite(value):
            return
        self._writer.add_scalar(tag, value, step)

    def scalars(self, values: dict[str, float], step: int, prefix: str = "") -> None:
        for tag, value in values.items():
            self.scalar(f"{prefix}{tag}" if prefix else tag, value, step)

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

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
        if self._hparam_writer is not None:
            self._hparam_writer.close()

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
