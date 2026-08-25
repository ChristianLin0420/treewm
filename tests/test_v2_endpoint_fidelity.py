import numpy as np

from treewm.data.future_sets import FutureSetBuilder, FutureSetConfig
from treewm.data.ogbench_dataset import TrajectoryIndex


def test_future_item_carries_explicit_task_metric_dimensions():
    obs = np.arange(36, dtype=np.float32).reshape(12, 3)
    act = np.zeros((12, 1), dtype=np.float32)
    terminal = np.zeros(12, dtype=np.float32)
    terminal[[5, 11]] = 1
    index = TrajectoryIndex.from_terminals(terminal)
    cfg = FutureSetConfig(
        num_neighbors=1,
        horizons=(1,),
        h_max=1,
        metric_mode="rms_v2",
        retrieval_radius=10,
        displacement_threshold=0,
        cluster_threshold=1,
        max_modes=1,
    )
    builder = FutureSetBuilder(
        obs,
        act,
        index,
        cfg,
        xy_dims=(0, 2),
        task_metric_dims=(0, 2),
    )
    item = builder.build(0)
    np.testing.assert_array_equal(item["task_metric_dims"], np.array([0, 2]))
    assert item["fut_metric_endpoint"].shape[-1] == 2
