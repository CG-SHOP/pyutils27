import numpy as np
import pytest
from conftest import footprint

from cgshop2027_pyutils.schemas import CGSHOP2027Instance, CGSHOP2027Solution
from cgshop2027_pyutils.verify import SolutionValidator, check_for_errors

UNIT_CUTTER = {"x": [0, 1, 1, 0], "y": [0, 0, 1, 1]}
BIG_CUTTER = {"x": [0, 2, 2, 0], "y": [0, 0, 2, 2]}
# An L, so that no symmetry can hide a misplaced anchor; every center below is
# a vertex of it, which the schema accepts because the cutter is closed.
L_CUTTER = {"x": [0, 2, 2, 1, 1, 0], "y": [0, 0, 1, 1, 2, 2]}
L_CUTTER_CELLS = [(0, 0), (0, 1), (1, 0)]
L_CENTERS = [(0, 0), (1, 1), (2, 0), (0, 2), (1, 2)]


def make_instance(size: int = 3, cutter=UNIT_CUTTER, center=(0, 0), cutters: int = 1):
    return CGSHOP2027Instance(
        instance_uid="demo",
        region_to_cover={
            "outer_boundary": {"x": [0, size, size, 0], "y": [0, 0, size, size]},
            "inner_boundaries": [],
        },
        cutter=cutter,
        cutter_center=center,
        number_of_cutters=cutters,
    )


def make_solution(*tours, instance_uid: str = "demo"):
    return CGSHOP2027Solution(instance_uid=instance_uid, tours=list(tours))


def tour(*points):
    """A closed tour from its corner points, in the schema's coordinate lists."""
    return {"x": [x for x, _ in points], "y": [y for _, y in points]}


# Boustrophedon over a 3x3 region, coming home along the row it started on.
FULL_SWEEP = tour((0, 0), (2, 0), (2, 1), (0, 1), (0, 2), (2, 2), (2, 0))
PARKED = tour((7, 7))


# --------------------------------------------------------------------------
# bookkeeping
# --------------------------------------------------------------------------


def test_a_full_sweep_is_accepted():
    assert check_for_errors(make_instance(), make_solution(FULL_SWEEP)) == []


def test_solution_for_another_instance_is_rejected_outright():
    errors = check_for_errors(
        make_instance(), make_solution(FULL_SWEEP, instance_uid="other")
    )
    assert len(errors) == 1
    assert "'other'" in errors[0]


def test_more_tours_than_cutters_is_rejected():
    errors = check_for_errors(
        make_instance(cutters=1), make_solution(FULL_SWEEP, PARKED)
    )
    assert errors == ["The solution has 2 tours, but the instance calls for exactly 1."]


def test_fewer_tours_than_cutters_is_rejected():
    """Every cutter needs a tour, even one that leaves it where it is."""
    errors = check_for_errors(make_instance(cutters=2), make_solution(FULL_SWEEP))
    assert errors == ["The solution has 1 tour, but the instance calls for exactly 2."]


def test_an_idle_cutter_is_given_as_a_tour_of_one_point():
    instance = make_instance(cutters=2)
    assert check_for_errors(instance, make_solution(FULL_SWEEP, PARKED)) == []


# --------------------------------------------------------------------------
# coverage
# --------------------------------------------------------------------------


INCOMPLETE = tour((0, 0), (2, 0), (2, 1), (0, 1))  # the top row stays untouched


def test_an_uncovered_cell_is_reported_by_name():
    errors = check_for_errors(make_instance(), make_solution(INCOMPLETE))
    assert len(errors) == 1
    assert "(0, 2)" in errors[0]
    assert "3 cells" in errors[0]


def test_uncovered_cells_are_exposed_for_inspection():
    validator = SolutionValidator(make_instance())
    uncovered = validator.uncovered_cells(make_solution(INCOMPLETE))
    assert sorted(uncovered.cells()) == [(0, 2), (1, 2), (2, 2)]
    assert not len(validator.uncovered_cells(make_solution(FULL_SWEEP)))


def test_two_cutters_may_share_the_work():
    lower = tour((0, 0), (2, 0), (2, 1), (0, 1))
    upper = tour((0, 2), (2, 2))
    assert check_for_errors(make_instance(cutters=2), make_solution(lower, upper)) == []


def test_cutters_may_sweep_the_same_cells():
    """There is no collision rule: two cutters may share a tour, even in full."""
    assert (
        check_for_errors(
            make_instance(cutters=2), make_solution(FULL_SWEEP, FULL_SWEEP)
        )
        == []
    )


def test_a_cutter_may_leave_the_region():
    """There is no containment rule: sweeping outside the region is allowed."""
    detour = tour(
        (0, 0),
        (-3, 0),  # wanders off to the west and comes back
        (2, 0),
        (2, 1),
        (0, 1),
        (0, 2),
        (2, 2),
        (2, 0),
    )
    assert check_for_errors(make_instance(), make_solution(detour)) == []


def test_a_larger_cutter_covers_more_per_step():
    """A 2x2 cutter anchored at its lower-left corner covers 4x4 in a loop."""
    instance = make_instance(size=4, cutter=BIG_CUTTER)
    loop = tour((0, 0), (2, 0), (2, 2), (0, 2))
    assert check_for_errors(instance, make_solution(loop)) == []


def test_a_stationary_cutter_covers_only_its_own_footprint():
    instance = make_instance(size=2, cutter=BIG_CUTTER)
    assert check_for_errors(instance, make_solution(tour((0, 0)))) == []


# --------------------------------------------------------------------------
# the anchor is the cutter's center
# --------------------------------------------------------------------------


@pytest.mark.parametrize("center", L_CENTERS)
def test_a_cutter_anchored_at_its_center_sits_where_the_instance_draws_it(center):
    """
    A tour's points are where the cutter's *center* goes, so parking the anchor
    on `cutter_center` has to reproduce the cutter in exactly the coordinates
    the instance spells out, whatever the center happens to be.
    """
    instance = make_instance(size=3, cutter=L_CUTTER, center=center)
    swept = SolutionValidator(instance).swept_cells(make_solution(tour(center)))
    assert sorted(swept.cells()) == L_CUTTER_CELLS


@pytest.mark.parametrize("center", L_CENTERS)
def test_moving_the_anchor_translates_the_footprint(center):
    """The center is an offset, not a second degree of freedom."""
    instance = make_instance(size=3, cutter=L_CUTTER, center=center)
    validator = SolutionValidator(instance)
    anchor = (center[0] + 1, center[1] + 1)
    swept = validator.swept_cells(make_solution(tour(anchor)))
    assert sorted(swept.cells()) == [(x + 1, y + 1) for x, y in L_CUTTER_CELLS]


@pytest.mark.parametrize("center", L_CENTERS)
def test_the_center_does_not_change_what_a_cutter_covers_relative_to_itself(center):
    """
    Two instances that differ only in `cutter_center` describe the same cutter,
    so the same tour shifted by the difference of the centers must sweep the
    same cells. This fails for any offset applied with the wrong sign.
    """
    loop = ((1, 1), (3, 1), (3, 3), (1, 3))
    baseline = SolutionValidator(make_instance(size=5, cutter=L_CUTTER, center=(0, 0)))
    shifted = SolutionValidator(make_instance(size=5, cutter=L_CUTTER, center=center))
    assert sorted(baseline.swept_cells(make_solution(tour(*loop))).cells()) == sorted(
        shifted.swept_cells(
            make_solution(tour(*((x + center[0], y + center[1]) for x, y in loop)))
        ).cells()
    )


def test_holes_do_not_have_to_be_covered():
    instance = CGSHOP2027Instance(
        instance_uid="demo",
        region_to_cover={
            "outer_boundary": {"x": [0, 3, 3, 0], "y": [0, 0, 3, 3]},
            "inner_boundaries": [{"x": [1, 1, 2, 2], "y": [1, 2, 2, 1]}],
        },
        cutter=UNIT_CUTTER,
        cutter_center=(0, 0),
        number_of_cutters=1,
    )
    # never enters the hole at (1, 1)
    ring = tour((0, 0), (2, 0), (2, 2), (0, 2))
    assert check_for_errors(instance, make_solution(ring)) == []


# --------------------------------------------------------------------------
# cross-check against a naive reference
# --------------------------------------------------------------------------


def naive_uncovered(validator, solution) -> set[tuple[int, int]]:
    """Marks every cutter cell at every unit of travel."""
    cells = footprint(validator.instance)
    swept = set()
    for cutter in solution.tours:
        for x, y in cutter.positions():
            swept.update((x + dx, y + dy) for dx, dy in cells)
    return set(validator.region.cells()) - swept


def random_cutter(rng):
    width, height = int(rng.integers(1, 4)), int(rng.integers(1, 4))
    if rng.random() < 0.5 and width > 1 and height > 1:
        notch_x, notch_y = int(rng.integers(1, width)), int(rng.integers(1, height))
        return (
            {
                "x": [0, width, width, notch_x, notch_x, 0],
                "y": [0, 0, notch_y, notch_y, height, height],
            },
            (int(rng.integers(0, notch_x + 1)), int(rng.integers(0, notch_y + 1))),
        )
    return (
        {"x": [0, width, width, 0], "y": [0, 0, height, height]},
        (int(rng.integers(0, width + 1)), int(rng.integers(0, height + 1))),
    )


def random_tour(rng, span):
    """
    A random closed tour: an axis-parallel walk, brought home in at most two
    legs. Points that repeat the one before them -- which the closing edge
    extends to a last point repeating the first -- are dropped afterwards,
    since dropping a duplicate leaves the geometry, and hence every edge,
    exactly as it was.
    """
    start = (int(rng.integers(-1, span)), int(rng.integers(-1, span)))
    points, (x, y) = [start], start
    for _ in range(int(rng.integers(0, 5))):
        distance = int(rng.integers(1, 4)) * (1 if rng.random() < 0.5 else -1)
        x, y = (x + distance, y) if rng.random() < 0.5 else (x, y + distance)
        points.append((x, y))
    points.append((start[0], y))
    points.append(start)
    trimmed = [
        point
        for index, point in enumerate(points)
        if index == 0 or point != points[index - 1]
    ]
    while len(trimmed) > 1 and trimmed[-1] == trimmed[0]:
        trimmed.pop()
    return tour(*trimmed)


@pytest.mark.parametrize("seed", range(40))
def test_matches_a_naive_reference_implementation(seed: int):
    """
    The verifier sweeps a whole edge at a time and dilates once, so it is
    compared against an implementation that does neither: one that stamps the
    cutter at every single unit of travel.
    """
    rng = np.random.default_rng(seed)
    span = int(rng.integers(2, 5))
    cutter, center = random_cutter(rng)
    instance = make_instance(size=span, cutter=cutter, center=center)
    solution = make_solution(
        *(random_tour(rng, span) for _ in range(int(rng.integers(1, 4))))
    )
    validator = SolutionValidator(instance)
    assert set(validator.uncovered_cells(solution).cells()) == naive_uncovered(
        validator, solution
    )


def test_larger_simple_instance_with_hole():
    instance = CGSHOP2027Instance(
        instance_uid="large_single_hole",
        region_to_cover={
            "outer_boundary": {"x": [128, 384, 384, 128], "y": [512, 512, 768, 768]},
            "inner_boundaries": [
                {"x": [160, 160, 352, 352], "y": [544, 736, 736, 544]}
            ],
        },
        cutter={"x": [0, 32, 32, 16, 16, 0], "y": [0, 0, 16, 16, 32, 32]},
        cutter_center=[16, 16],
        number_of_cutters=1,
    )

    solution = CGSHOP2027Solution(
        instance_uid="large_single_hole",
        meta={},
        tours=[tour((144, 528), (144, 752), (368, 752), (368, 528))],
    )

    validator = SolutionValidator(instance)
    errors = validator.check_for_errors(solution)
    assert len(errors) == 1
    assert "region is not covered" in errors[0]

    solution2 = CGSHOP2027Solution(
        instance_uid="large_single_hole",
        meta={},
        tours=[
            tour((144, 528), (144, 752), (384, 752), (368, 752), (368, 528)),
        ],
    )
    assert validator.check_for_errors(solution2) == []


# --------------------------------------------------------------------------
# integration
# --------------------------------------------------------------------------


def test_validator_is_reusable_across_solutions():
    validator = SolutionValidator(make_instance())
    assert validator.check_for_errors(make_solution(FULL_SWEEP)) == []
    assert validator.check_for_errors(make_solution(tour((0, 0)))) != []
    assert validator.check_for_errors(make_solution(FULL_SWEEP)) == []


def test_verifies_instance_and_solution_read_from_files(tmp_path):
    from cgshop2027_pyutils.io import read_instance, read_solution

    instance = make_instance()
    solution = make_solution(FULL_SWEEP)
    instance_path = tmp_path / "demo.instance.json"
    solution_path = tmp_path / "demo.solution.json"
    instance_path.write_text(instance.model_dump_json())
    solution_path.write_text(solution.model_dump_json())
    assert (
        check_for_errors(read_instance(instance_path), read_solution(solution_path))
        == []
    )
