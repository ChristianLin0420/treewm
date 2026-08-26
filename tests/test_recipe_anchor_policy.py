"""Contracts for consuming the full immutable future-recipe anchor union."""

from types import SimpleNamespace

import numpy as np
import pytest

from treewm.data.ogbench_dataset import apply_recipe_anchor_policy


def test_published_union_replaces_both_selected_seed_slices():
    train = SimpleNamespace(anchors=np.asarray([2, 7], dtype=np.int64))
    val = SimpleNamespace(anchors=np.asarray([3], dtype=np.int64))
    train_recipe = SimpleNamespace(anchors=np.asarray([1, 2, 7, 9], dtype=np.int64))
    val_recipe = SimpleNamespace(anchors=np.asarray([3, 4], dtype=np.int64))

    apply_recipe_anchor_policy(
        train, val, train_recipe, val_recipe, "published_union"
    )

    np.testing.assert_array_equal(train.anchors, [1, 2, 7, 9])
    np.testing.assert_array_equal(val.anchors, [3, 4])


def test_selected_seed_is_backward_compatible_and_unknown_policy_fails():
    train = SimpleNamespace(anchors=np.asarray([2, 7], dtype=np.int64))
    val = SimpleNamespace(anchors=np.asarray([3], dtype=np.int64))
    recipe = SimpleNamespace(anchors=np.asarray([1, 2, 3, 7], dtype=np.int64))

    apply_recipe_anchor_policy(train, val, recipe, recipe, "selected_seed")
    np.testing.assert_array_equal(train.anchors, [2, 7])
    np.testing.assert_array_equal(val.anchors, [3])
    with pytest.raises(ValueError, match="unsupported recipe anchor policy"):
        apply_recipe_anchor_policy(train, val, recipe, recipe, "all")
