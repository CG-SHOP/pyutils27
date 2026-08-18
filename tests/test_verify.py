import numpy as np
import pytest
from shapely.affinity import translate

from cgshop2027_pyutils.schemas import CGSHOP2027Instance, CGSHOP2027Solution
from cgshop2027_pyutils.schemas.solution import _DIRECTION_DELTAS
from cgshop2027_pyutils.verify import SolutionValidator, check_for_errors

UNIT_CUTTER = {"x": [0, 1, 1, 0], "y": [0, 0, 1, 1]}
BIG_CUTTER = {"x": [0, 2, 2, 0], "y": [0, 0, 2, 2]}


def make_instance(size: int = 3, cutter=UNIT_CUTTER, center=(0, 0), cutters: int = 4):
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


def make_solution(*paths, instance_uid: str = "demo"):
    return CGSHOP2027Solution(instance_uid=instance_uid, cutters=list(paths))


def path(start, *steps):
    return {"start": start, "steps": list(steps)}


FULL_SWEEP = path(
    (0, 0), ("E", 2), ("N", 1), ("W", 2), ("N", 1), ("E", 2), ("W", 2), ("S", 2)
)


def overlaps(errors: list[str]) -> list[str]:
    return [error for error in errors if "overlap" in error]


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


def test_using_more_cutters_than_available_is_rejected():
    instance = make_instance(cutters=1)
    errors = check_for_errors(instance, make_solution(FULL_SWEEP, FULL_SWEEP))
    assert any("only 1 are available" in error for error in errors)


def test_using_fewer_cutters_than_available_is_fine():
    assert check_for_errors(make_instance(cutters=4), make_solution(FULL_SWEEP)) == []


# --------------------------------------------------------------------------
# coverage
# --------------------------------------------------------------------------


def test_an_uncovered_cell_is_reported_by_name():
    incomplete = path((0, 0), ("E", 2), ("N", 1), ("W", 2), ("S", 1))
    errors = check_for_errors(make_instance(), make_solution(incomplete))
    assert len(errors) == 1
    assert "(0, 2)" in errors[0]
    assert "3 cells" in errors[0]


def test_uncovered_cells_are_exposed_for_inspection():
    validator = SolutionValidator(make_instance())
    incomplete = path((0, 0), ("E", 2), ("N", 1), ("W", 2), ("S", 1))
    uncovered = validator.uncovered_cells(make_solution(incomplete))
    assert sorted(uncovered.cells()) == [(0, 2), (1, 2), (2, 2)]
    assert not len(validator.uncovered_cells(make_solution(FULL_SWEEP)))


def test_two_cutters_may_share_the_work():
    lower = path((0, 0), ("E", 2), ("N", 1), ("W", 2), ("S", 1))
    upper = path((0, 2), ("E", 2), ("W", 2))
    assert check_for_errors(make_instance(), make_solution(lower, upper)) == []


def test_a_cutter_may_leave_the_region():
    """There is no containment rule: sweeping outside the region is allowed."""
    detour = path(
        (0, 0),
        ("W", 3),
        ("E", 3),  # wanders off to the west and comes back
        ("E", 2),
        ("N", 1),
        ("W", 2),
        ("N", 1),
        ("E", 2),
        ("W", 2),
        ("S", 2),
    )
    assert check_for_errors(make_instance(), make_solution(detour)) == []


def test_a_larger_cutter_covers_more_per_step():
    """A 2x2 cutter anchored at its lower-left corner covers 4x4 in a loop."""
    instance = make_instance(size=4, cutter=BIG_CUTTER)
    loop = path((0, 0), ("E", 2), ("N", 2), ("W", 2), ("S", 2))
    assert check_for_errors(instance, make_solution(loop)) == []


def test_a_stationary_cutter_covers_only_its_own_footprint():
    instance = make_instance(size=2, cutter=BIG_CUTTER)
    assert check_for_errors(instance, make_solution(path((0, 0)))) == []


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
    ring = path(
        (0, 0), ("E", 2), ("N", 2), ("W", 2), ("S", 2)
    )  # never enters the hole at (1, 1)
    assert check_for_errors(instance, make_solution(ring)) == []


# --------------------------------------------------------------------------
# collisions
# --------------------------------------------------------------------------


def test_cutters_on_the_same_cell_at_the_same_time_collide():
    errors = check_for_errors(
        make_instance(),
        make_solution(path((0, 0), ("I", 2)), path((0, 0), ("I", 2))),
    )
    assert overlaps(errors) == ["Cutters #0 and #1 overlap at time 0."]


def test_cutters_touching_along_an_edge_do_not_collide():
    """Only positive-area overlap counts; sharing a boundary is legal."""
    errors = check_for_errors(
        make_instance(),
        make_solution(path((0, 0), ("I", 2)), path((1, 0), ("I", 2))),
    )
    assert overlaps(errors) == []


def test_visiting_the_same_cell_at_different_times_is_fine():
    errors = check_for_errors(
        make_instance(),
        make_solution(
            path((0, 0), ("E", 1), ("W", 1)),  # at (1, 0) at time 1
            path((2, 0), ("I", 2), ("W", 1), ("E", 1)),  # at (1, 0) at time 3
        ),
    )
    assert overlaps(errors) == []


def test_a_head_on_swap_collides_although_both_snapshots_are_clean():
    """
    Two unit cutters closing head-on occupy the same square halfway through the
    move; at both integer times they only touch. Sampling the trajectory at
    integer times alone would miss this.
    """
    left = path((0, 0), ("E", 1), ("W", 1))
    right = path((1, 0), ("W", 1), ("E", 1))
    errors = check_for_errors(make_instance(), make_solution(left, right))
    assert overlaps(errors) == [
        "Cutters #0 and #1 overlap while moving between time 0 and 1."
    ]


def test_a_diagonal_pass_collides():
    """
    One cutter moving north past another moving east sweeps through it, even
    though the two never share a cell at an integer time.
    """
    going_north = path((0, 0), ("N", 1), ("S", 1))
    going_east = path((0, 1), ("E", 1), ("W", 1))
    errors = check_for_errors(make_instance(), make_solution(going_north, going_east))
    assert overlaps(errors) == [
        "Cutters #0 and #1 overlap while moving between time 0 and 1."
    ]


def test_cutters_moving_in_convoy_never_collide():
    convoy = [path((0, 0), ("E", 2), ("W", 2)), path((2, 0), ("E", 2), ("W", 2))]
    assert overlaps(check_for_errors(make_instance(), make_solution(*convoy))) == []


def test_a_finished_cutter_stays_parked_and_can_be_hit():
    """
    Trajectories of different length are padded with idling: a cutter that is
    done sits at its start position and still blocks the others.
    """
    parked = path((0, 0))
    passing = path((3, 0), ("W", 3), ("E", 3))
    errors = check_for_errors(make_instance(), make_solution(parked, passing))
    assert overlaps(errors) == ["Cutters #0 and #1 overlap at time 3."]


def test_a_finished_cutter_parked_out_of_the_way_is_harmless():
    parked = path((9, 9))
    errors = check_for_errors(make_instance(), make_solution(parked, FULL_SWEEP))
    assert overlaps(errors) == []


def test_the_earliest_collision_is_reported():
    """
    This pair collides twice: it swaps during the first move and then both sit
    on (1, 0) at time 2. Only the earlier of the two is reported, and a swap
    counts as earlier than an overlap at the time the move ends.
    """
    errors = check_for_errors(
        make_instance(),
        make_solution(
            path((0, 0), ("E", 1), ("I", 1), ("W", 1)),
            path((1, 0), ("W", 1), ("E", 1)),
        ),
    )
    assert overlaps(errors) == [
        "Cutters #0 and #1 overlap while moving between time 0 and 1."
    ]


def test_every_colliding_pair_is_reported():
    errors = overlaps(
        check_for_errors(
            make_instance(),
            make_solution(*(path((0, 0), ("I", 1)) for _ in range(3))),
        )
    )
    assert len(errors) == 3


# --------------------------------------------------------------------------
# cross-check against a naive reference
# --------------------------------------------------------------------------


def naive_uncovered(validator, solution) -> set[tuple[int, int]]:
    """Marks every cutter cell at every integer time."""
    footprint = list(validator.cutter.cells())
    swept = set()
    for cutter in solution.cutters:
        for x, y in cutter.positions():
            swept.update((x + dx, y + dy) for dx, dy in footprint)
    return set(validator.region.cells()) - swept


def naive_collides(validator, solution) -> bool:
    """
    Samples the continuous motion. The offset between two cutters crosses a
    cell boundary only at multiples of 1/2, so testing quarter steps decides
    the question exactly.
    """
    shape = validator.cutter.to_polygons()
    tracks = [list(cutter.positions()) for cutter in solution.cutters]
    span = max((len(track) for track in tracks), default=0)
    tracks = [track + [track[-1]] * (span - len(track)) for track in tracks]
    for time in range(span):
        for i, track_i in enumerate(tracks):
            for track_j in tracks[:i]:
                for fraction in (0.0, 0.25, 0.5, 0.75):
                    placements = [
                        translate(
                            shape,
                            track[time][0]
                            + fraction
                            * (track[min(time + 1, span - 1)][0] - track[time][0]),
                            track[time][1]
                            + fraction
                            * (track[min(time + 1, span - 1)][1] - track[time][1]),
                        )
                        for track in (track_i, track_j)
                    ]
                    if placements[0].intersection(placements[1]).area > 1e-9:
                        return True
    return False


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


def random_path(rng, span):
    start = (int(rng.integers(-1, span)), int(rng.integers(-1, span)))
    steps, dx, dy = [], 0, 0
    for _ in range(int(rng.integers(0, 5))):
        direction = "NESWI"[int(rng.integers(0, 5))]
        units = int(rng.integers(1, 4))
        steps.append((direction, units))
        dx += _DIRECTION_DELTAS[direction][0] * units
        dy += _DIRECTION_DELTAS[direction][1] * units
    if dx:
        steps.append(("W" if dx > 0 else "E", abs(dx)))
    if dy:
        steps.append(("S" if dy > 0 else "N", abs(dy)))
    return {"start": start, "steps": steps}


@pytest.mark.parametrize("seed", range(40))
def test_matches_a_naive_reference_implementation(seed: int):
    """
    Both checks skip the time axis wherever they can, so they are compared
    against implementations that do not: coverage by stamping the cutter at
    every time step, collisions by sampling the motion in between.
    """
    rng = np.random.default_rng(seed)
    span = int(rng.integers(2, 5))
    cutter, center = random_cutter(rng)
    instance = make_instance(size=span, cutter=cutter, center=center)
    solution = make_solution(
        *(random_path(rng, span) for _ in range(int(rng.integers(1, 4))))
    )
    validator = SolutionValidator(instance)
    assert set(validator.uncovered_cells(solution).cells()) == naive_uncovered(
        validator, solution
    )
    assert bool(overlaps(validator.check_for_errors(solution))) == naive_collides(
        validator, solution
    )


# --------------------------------------------------------------------------
# integration
# --------------------------------------------------------------------------


def test_validator_is_reusable_across_solutions():
    validator = SolutionValidator(make_instance())
    assert validator.check_for_errors(make_solution(FULL_SWEEP)) == []
    assert validator.check_for_errors(make_solution(path((0, 0)))) != []
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
