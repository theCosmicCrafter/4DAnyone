"""Unit tests for view plan resolution and group sizing."""

from fdanyone.errors import ConfigurationError
from fdanyone.views import resolve_view_plan, VALID_VIEWS_PER_GROUP


def test_valid_views_per_group():
    assert 2 in VALID_VIEWS_PER_GROUP
    assert 3 in VALID_VIEWS_PER_GROUP
    assert 4 in VALID_VIEWS_PER_GROUP
    assert 6 in VALID_VIEWS_PER_GROUP


def test_group_size_2():
    plan = resolve_view_plan(views_per_layer=24, views_per_group=2)
    assert plan.views_per_group == 2
    assert plan.groups_per_layer == 12
    assert plan.num_target_views == 24


def test_group_size_3():
    plan = resolve_view_plan(views_per_layer=24, views_per_group=3)
    assert plan.views_per_group == 3
    assert plan.groups_per_layer == 8
    assert plan.num_target_views == 24


def test_group_size_auto():
    plan = resolve_view_plan(views_per_layer=24, views_per_group="auto")
    assert plan.views_per_group == 6


def test_multi_pitch_layers():
    plan = resolve_view_plan(views_per_layer=12, layer_pitches=[-10, 15, 35], views_per_group=3)
    assert plan.num_layers == 3
    assert plan.num_target_views == 36
    assert plan.num_groups == 12


def test_invalid_group_size():
    try:
        resolve_view_plan(views_per_layer=24, views_per_group=5)
    except ConfigurationError:
        pass
    else:
        assert False, "Expected ConfigurationError for group size 5"
