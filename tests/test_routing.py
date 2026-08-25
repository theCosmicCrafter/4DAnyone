"""Unit tests for Target-Context Routing (TCR) permutation algorithms."""

from fdanyone.model.routing import routing_steps, view_groups


def test_view_groups_circular():
    groups = view_groups(num_views=6, group_size=2, offset=0, circular=True)
    assert groups == ((0, 1), (2, 3), (4, 5))

    groups_shifted = view_groups(num_views=6, group_size=2, offset=1, circular=True)
    assert groups_shifted == ((1, 2), (3, 4), (5, 0))


def test_routing_steps_tcr():
    # 24 steps, 24 views per layer, group size 2
    steps = routing_steps(
        views_per_layer=24,
        num_layers=1,
        group_size=2,
        num_steps=24,
        enable_tcr=True,
        circular=True,
    )
    assert len(steps) == 24
    for group_list in steps:
        assert len(group_list) == 12
        for g in group_list:
            assert len(g) == 2


def test_routing_steps_group_3():
    steps = routing_steps(
        views_per_layer=24,
        num_layers=1,
        group_size=3,
        num_steps=24,
        enable_tcr=True,
        circular=True,
    )
    assert len(steps) == 24
    for group_list in steps:
        assert len(group_list) == 8
        for g in group_list:
            assert len(g) == 3
