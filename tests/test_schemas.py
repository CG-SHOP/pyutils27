import pytest
from pydantic import ValidationError
from shapely.geometry import LinearRing

from cgshop2027_pyutils.schemas.instance import (
    CGSHOP2027Instance,
    PointSequence,
    PolyominoWithHoles,
)
from cgshop2027_pyutils.schemas.solution import CGSHOP2027Solution, CutterPath

# Rectilinear, integer-coordinate boundaries shared by the tests below.
SQUARE_CCW = {"x": [0, 4, 4, 0], "y": [0, 0, 4, 4]}
SQUARE_CW = {"x": [0, 0, 4, 4], "y": [0, 4, 4, 0]}
HOLE_CW = {"x": [1, 1, 3, 3], "y": [1, 3, 3, 1]}
HOLE_CCW = {"x": [1, 3, 3, 1], "y": [1, 1, 3, 3]}
L_SHAPE_CCW = {"x": [0, 2, 2, 1, 1, 0], "y": [0, 0, 1, 1, 2, 2]}


# --------------------------------------------------------------------------
# PointSequence: orientation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "boundary", [SQUARE_CCW, SQUARE_CW, HOLE_CW, HOLE_CCW, L_SHAPE_CCW]
)
def test_is_ccw_agrees_with_shapely(boundary: dict[str, list[int]]):
    """Our orientation test must match the reference implementation."""
    sequence = PointSequence(**boundary)
    expected = LinearRing(zip(boundary["x"], boundary["y"], strict=True)).is_ccw
    assert sequence.is_ccw() is expected


@pytest.mark.parametrize(("boundary", "ccw"), [(SQUARE_CCW, True), (SQUARE_CW, False)])
def test_is_ccw_of_known_orientation(boundary: dict[str, list[int]], ccw: bool):
    assert PointSequence(**boundary).is_ccw() is ccw


def test_reversing_the_sequence_flips_the_orientation():
    sequence = PointSequence(**L_SHAPE_CCW)
    reversed_sequence = PointSequence(
        x=list(reversed(L_SHAPE_CCW["x"])), y=list(reversed(L_SHAPE_CCW["y"]))
    )
    assert sequence.is_ccw() is not reversed_sequence.is_ccw()


# --------------------------------------------------------------------------
# PointSequence: accepted input
# --------------------------------------------------------------------------


def test_point_sequence_accepts_distinct_points():
    seq = PointSequence(x=[0, 1, 1, 0], y=[0, 0, 1, 1])
    assert seq.x == [0, 1, 1, 0]
    assert seq.y == [0, 0, 1, 1]


def test_point_sequence_accepts_repeated_coordinates():
    # The individual coordinate lists may repeat values as long as the
    # resulting points are distinct.
    seq = PointSequence(**SQUARE_CCW)
    assert seq.x.count(0) == 2


def test_point_sequence_accepts_non_convex_boundary():
    PointSequence(**L_SHAPE_CCW)


def test_point_sequence_survives_a_json_round_trip():
    sequence = PointSequence(**L_SHAPE_CCW)
    assert PointSequence.model_validate_json(sequence.model_dump_json()) == sequence


# --------------------------------------------------------------------------
# PointSequence: rejected input
# --------------------------------------------------------------------------


def test_point_sequence_rejects_duplicate_points():
    with pytest.raises(ValidationError, match=r"point \(0, 0\) at index 3"):
        PointSequence(x=[0, 1, 1, 0], y=[0, 0, 1, 0])


def test_point_sequence_rejects_mismatched_lengths():
    with pytest.raises(ValidationError, match="same length"):
        PointSequence(x=[0, 1], y=[0])


@pytest.mark.parametrize(
    ("x", "y"), [([], []), ([0], [0]), ([0, 1, 1], [0, 0, 1])], ids=["0", "1", "3"]
)
def test_point_sequence_rejects_too_few_points(x: list[int], y: list[int]):
    with pytest.raises(ValidationError, match="at least 4 points"):
        PointSequence(x=x, y=y)


def test_point_sequence_rejects_diagonal_edge():
    with pytest.raises(
        ValidationError, match=r"from \(0, 0\) to \(2, 1\) is not ortho"
    ):
        PointSequence(x=[0, 2, 2, 0], y=[0, 1, 2, 2])


def test_point_sequence_rejects_diagonal_closing_edge():
    """The implicit edge from the last back to the first point is checked too."""
    with pytest.raises(
        ValidationError, match=r"from \(2, 1\) to \(0, 0\) is not ortho"
    ):
        PointSequence(x=[0, 0, 2, 2], y=[0, 2, 2, 1])


def test_point_sequence_rejects_self_intersection():
    # Rectilinear figure-eight: the edge (1,2)->(1,-1) crosses the edge (0,0)->(2,0).
    with pytest.raises(ValidationError, match="self-intersecting"):
        PointSequence(x=[0, 2, 2, 1, 1, 0], y=[0, 0, 2, 2, -1, -1])


def test_point_sequence_rejects_pinch_point_as_duplicate():
    """
    Two cells joined only at a corner repeat the shared vertex, so the
    duplicate check rejects them before the self-intersection check runs.
    """
    with pytest.raises(ValidationError, match=r"point \(1, 1\) at index 6"):
        PointSequence(x=[0, 1, 1, 2, 2, 1, 1, 0], y=[0, 0, 1, 1, 2, 2, 1, 1])


def test_point_sequence_rejects_fractional_coordinates():
    with pytest.raises(ValidationError, match="valid integer"):
        PointSequence(x=[0, 1.5, 1, 0], y=[0, 0, 1, 1])


# --------------------------------------------------------------------------
# PolyominoWithHoles
# --------------------------------------------------------------------------


def test_polyomino_accepts_boundary_without_holes():
    poly = PolyominoWithHoles(outer_boundary=SQUARE_CCW, inner_boundaries=[])
    assert poly.outer_boundary.is_ccw()
    assert poly.inner_boundaries == []


def test_polyomino_accepts_a_clockwise_hole():
    poly = PolyominoWithHoles(outer_boundary=SQUARE_CCW, inner_boundaries=[HOLE_CW])
    assert not poly.inner_boundaries[0].is_ccw()


def test_polyomino_accepts_already_parsed_point_sequences():
    poly = PolyominoWithHoles(
        outer_boundary=PointSequence(**SQUARE_CCW),
        inner_boundaries=[PointSequence(**HOLE_CW)],
    )
    assert poly.outer_boundary == PointSequence(**SQUARE_CCW)


def test_polyomino_survives_a_json_round_trip():
    poly = PolyominoWithHoles(outer_boundary=SQUARE_CCW, inner_boundaries=[HOLE_CW])
    assert PolyominoWithHoles.model_validate_json(poly.model_dump_json()) == poly


def test_polyomino_rejects_clockwise_outer_boundary():
    with pytest.raises(ValidationError, match="outer boundary must be oriented count"):
        PolyominoWithHoles(outer_boundary=SQUARE_CW, inner_boundaries=[])


def test_polyomino_rejects_counter_clockwise_hole():
    with pytest.raises(ValidationError, match="inner boundary at index 0 must be"):
        PolyominoWithHoles(outer_boundary=SQUARE_CCW, inner_boundaries=[HOLE_CCW])


def test_polyomino_reports_the_index_of_the_offending_hole():
    with pytest.raises(ValidationError, match="inner boundary at index 1 must be"):
        PolyominoWithHoles(
            outer_boundary={"x": [0, 8, 8, 0], "y": [0, 0, 8, 8]},
            inner_boundaries=[HOLE_CW, {"x": [5, 7, 7, 5], "y": [5, 5, 7, 7]}],
        )


def test_polyomino_rejects_hole_outside_the_outer_boundary():
    with pytest.raises(ValidationError, match="is invalid"):
        PolyominoWithHoles(
            outer_boundary=SQUARE_CCW,
            inner_boundaries=[{"x": [5, 5, 6, 6], "y": [5, 6, 6, 5]}],
        )


def test_polyomino_rejects_hole_crossing_the_outer_boundary():
    with pytest.raises(ValidationError, match="is invalid"):
        PolyominoWithHoles(
            outer_boundary=SQUARE_CCW,
            inner_boundaries=[{"x": [3, 3, 5, 5], "y": [1, 3, 3, 1]}],
        )


def test_polyomino_rejects_overlapping_holes():
    with pytest.raises(ValidationError, match="is invalid"):
        PolyominoWithHoles(
            outer_boundary={"x": [0, 8, 8, 0], "y": [0, 0, 8, 8]},
            inner_boundaries=[
                {"x": [1, 1, 4, 4], "y": [1, 4, 4, 1]},
                {"x": [2, 2, 5, 5], "y": [2, 5, 5, 2]},
            ],
        )


def test_polyomino_requires_inner_boundaries():
    """
    `inner_boundaries` is deliberately required rather than defaulting to an
    empty list: participants write their own parsers, so a hole-free polyomino
    must still state `"inner_boundaries": []` explicitly.
    """
    with pytest.raises(ValidationError) as exc_info:
        PolyominoWithHoles(outer_boundary=SQUARE_CCW)
    error = exc_info.value.errors()[0]
    assert error["type"] == "missing"
    assert error["loc"] == ("inner_boundaries",)


@pytest.mark.parametrize(
    "dump_options",
    [{}, {"exclude_defaults": True}, {"exclude_unset": True}, {"exclude_none": True}],
    ids=["plain", "exclude_defaults", "exclude_unset", "exclude_none"],
)
def test_hole_free_polyomino_always_serializes_the_empty_hole_list(
    dump_options: dict[str, bool],
):
    """
    A hole-free polyomino must never be written out with the key omitted, no
    matter how it is dumped. Being a required field is what guarantees this;
    with a default the `exclude_defaults`/`exclude_unset` dumps would drop it.
    """
    poly = PolyominoWithHoles(outer_boundary=SQUARE_CCW, inner_boundaries=[])
    assert '"inner_boundaries":[]' in poly.model_dump_json(**dump_options)


def test_polyomino_reports_the_location_of_a_bad_outer_boundary():
    with pytest.raises(ValidationError) as exc_info:
        PolyominoWithHoles(
            outer_boundary={"x": [0, 1, 1, 0], "y": [0, 0, 1, 0]},
            inner_boundaries=[],
        )
    assert exc_info.value.errors()[0]["loc"] == ("outer_boundary",)


def test_polyomino_reports_the_location_of_a_bad_hole():
    with pytest.raises(ValidationError) as exc_info:
        PolyominoWithHoles(
            outer_boundary=SQUARE_CCW,
            inner_boundaries=[HOLE_CW, {"x": [1, 1, 1], "y": [1, 2, 3]}],
        )
    assert exc_info.value.errors()[0]["loc"] == ("inner_boundaries", 1)


# --------------------------------------------------------------------------
# CGSHOP2027Instance
# --------------------------------------------------------------------------

REGION = {
    "outer_boundary": {"x": [0, 8, 8, 0], "y": [0, 0, 8, 8]},
    "inner_boundaries": [],
}
CUTTER_CCW = {"x": [0, 2, 2, 0], "y": [0, 0, 2, 2]}  # 2x2 square, CCW
CUTTER_CW = {"x": [0, 0, 2, 2], "y": [0, 2, 2, 0]}  # the same square, reversed
UNIT_CUTTER_CCW = {"x": [0, 1, 1, 0], "y": [0, 0, 1, 1]}


def make_instance(**overrides):
    """Builds a valid instance, with individual fields replaceable per test."""
    fields = {
        "instance_uid": "demo",
        "region_to_cover": REGION,
        "cutter": CUTTER_CCW,
        "cutter_center": (1, 1),
        "number_of_cutters": 3,
    }
    fields.update(overrides)
    return CGSHOP2027Instance(**fields)


def test_instance_accepts_a_valid_instance():
    instance = make_instance()
    assert instance.content_type == "CGSHOP2027_Instance"
    assert instance.instance_uid == "demo"
    assert instance.cutter_center == (1, 1)
    assert instance.number_of_cutters == 3
    assert instance.region_to_cover.inner_boundaries == []


def test_instance_rejects_a_foreign_content_type():
    with pytest.raises(ValidationError, match="CGSHOP2027_Instance"):
        make_instance(content_type="CGSHOP2026_Instance")


@pytest.mark.parametrize(
    "field",
    ["instance_uid", "region_to_cover", "cutter", "cutter_center", "number_of_cutters"],
)
def test_instance_requires_every_problem_field(field: str):
    fields = {
        "instance_uid": "demo",
        "region_to_cover": REGION,
        "cutter": CUTTER_CCW,
        "cutter_center": (1, 1),
        "number_of_cutters": 3,
    }
    del fields[field]
    with pytest.raises(ValidationError) as exc_info:
        CGSHOP2027Instance(**fields)
    error = exc_info.value.errors()[0]
    assert error["type"] == "missing"
    assert error["loc"] == (field,)


def test_instance_survives_a_json_round_trip():
    instance = make_instance()
    assert (
        CGSHOP2027Instance.model_validate_json(instance.model_dump_json()) == instance
    )


# --- number_of_cutters ---


def test_instance_accepts_a_single_cutter():
    assert make_instance(number_of_cutters=1).number_of_cutters == 1


@pytest.mark.parametrize("count", [0, -1])
def test_instance_rejects_non_positive_number_of_cutters(count: int):
    with pytest.raises(ValidationError, match="greater than 0"):
        make_instance(number_of_cutters=count)


def test_instance_rejects_fractional_number_of_cutters():
    with pytest.raises(ValidationError, match="valid integer"):
        make_instance(number_of_cutters=2.5)


# --- cutter orientation ---


def test_instance_rejects_clockwise_cutter():
    with pytest.raises(ValidationError, match="cutter shape must be oriented count"):
        make_instance(cutter=CUTTER_CW)


def test_cutter_orientation_is_checked_before_the_center():
    """Both validators fail here; the orientation one is defined first."""
    with pytest.raises(ValidationError, match="cutter shape must be oriented count"):
        make_instance(cutter=CUTTER_CW, cutter_center=(99, 99))


# --- cutter_center ---


def test_instance_accepts_a_center_inside_the_cutter():
    assert make_instance(cutter_center=(1, 1)).cutter_center == (1, 1)


def test_instance_accepts_a_center_given_as_a_list():
    """JSON has no tuples, so a two-element array must be accepted."""
    assert make_instance(cutter_center=[1, 1]).cutter_center == (1, 1)


@pytest.mark.parametrize("center", [(3, 3), (-1, -1), (5, 1)])
def test_instance_rejects_a_center_outside_the_cutter(center: tuple[int, int]):
    with pytest.raises(ValidationError, match="center must be within the cutter"):
        make_instance(cutter_center=center)


@pytest.mark.parametrize(
    "center",
    [(0, 0), (2, 2), (1, 0), (0, 1)],
    ids=["corner", "opposite-corner", "edge", "edge"],
)
def test_instance_accepts_a_center_on_the_cutter_boundary(center: tuple[int, int]):
    """
    The cutter is treated as a closed region (`covers`, not `contains`), so a
    center on a corner or an edge is a legal anchor.
    """
    assert make_instance(cutter_center=center).cutter_center == center


@pytest.mark.parametrize("center", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_unit_cutter_admits_every_corner_as_center(center: tuple[int, int]):
    """
    A 1x1 cutter has no interior lattice point at all, so it is expressible
    only because boundary points are accepted.
    """
    assert make_instance(cutter=UNIT_CUTTER_CCW, cutter_center=center)


def test_instance_rejects_a_center_in_the_notch_of_a_concave_cutter():
    """
    Accepting the boundary must not degrade into accepting the bounding box:
    (2, 2) is the reflex corner's diagonal opposite and lies outside the L.
    """
    with pytest.raises(ValidationError, match="center must be within the cutter"):
        make_instance(cutter=L_SHAPE_CCW, cutter_center=(2, 2))


def test_instance_accepts_the_reflex_vertex_of_a_concave_cutter():
    assert make_instance(cutter=L_SHAPE_CCW, cutter_center=(1, 1)).cutter_center == (
        1,
        1,
    )


def test_instance_rejects_a_center_that_is_not_a_pair():
    with pytest.raises(ValidationError) as exc_info:
        make_instance(cutter_center=(1, 1, 1))
    assert exc_info.value.errors()[0]["loc"] == ("cutter_center",)


# --- nested error reporting ---


def test_instance_reports_the_location_of_a_bad_cutter():
    with pytest.raises(ValidationError) as exc_info:
        make_instance(cutter={"x": [0, 2, 2, 0], "y": [0, 1, 2, 2]})
    assert exc_info.value.errors()[0]["loc"] == ("cutter",)


def test_instance_reports_the_location_of_a_bad_region():
    with pytest.raises(ValidationError) as exc_info:
        make_instance(
            region_to_cover={
                "outer_boundary": {"x": [0, 0, 8, 8], "y": [0, 8, 8, 0]},
                "inner_boundaries": [],
            }
        )
    assert exc_info.value.errors()[0]["loc"] == ("region_to_cover",)


# --------------------------------------------------------------------------
# CGSHOP2027Solution
# --------------------------------------------------------------------------

# A closed L-shaped tour with an idle period in the middle. Cutters in a
# solution have to return to where they started, so the paths used here are
# loops; `OPEN_PATH` stays open to exercise `CutterPath` on its own, which
# carries no such requirement.
L_PATH = {
    "start": (3, 5),
    "steps": [("N", 3), ("I", 2), ("W", 1), ("E", 1), ("S", 3)],
}
STATIONARY_PATH = {"start": (0, 0), "steps": []}
OPEN_PATH = {"start": (3, 5), "steps": [("N", 3), ("I", 2), ("W", 1)]}


def make_solution(**overrides):
    """Builds a valid solution, with individual fields replaceable per test."""
    fields = {"instance_uid": "demo", "cutters": [L_PATH, STATIONARY_PATH]}
    fields.update(overrides)
    return CGSHOP2027Solution(**fields)


def test_solution_accepts_a_valid_solution():
    solution = make_solution()
    assert solution.content_type == "CGSHOP2027_Solution"
    assert solution.instance_uid == "demo"
    assert solution.meta == {}
    assert len(solution.cutters) == 2
    assert solution.cutters[0].steps == [
        ("N", 3),
        ("I", 2),
        ("W", 1),
        ("E", 1),
        ("S", 3),
    ]


def test_solution_accepts_a_stationary_cutter():
    """A cutter that never moves is expressed as an empty step list."""
    solution = make_solution(cutters=[STATIONARY_PATH])
    assert solution.cutters[0].steps == []
    assert solution.cutters[0].end == (0, 0)


def test_solution_accepts_no_cutters_at_all():
    """
    Whether the number of trajectories matches `number_of_cutters` can only be
    decided against the instance, so the solution schema stays permissive here.
    """
    assert make_solution(cutters=[]).cutters == []


def test_steps_may_be_given_as_lists():
    """JSON has no tuples, so the list form has to be accepted as well."""
    from_lists = make_solution(
        cutters=[{"start": [3, 5], "steps": [["N", 3], ["S", 3]]}]
    )
    from_tuples = make_solution(
        cutters=[{"start": (3, 5), "steps": [("N", 3), ("S", 3)]}]
    )
    assert from_lists == from_tuples


def test_solution_accepts_an_immediate_reversal():
    """Stepping back onto the previous position is a legal, if pointless, move."""
    path = make_solution(cutters=[{"start": (0, 0), "steps": [("N", 1), ("S", 1)]}])
    assert path.cutters[0].end == (0, 0)


def test_solution_accepts_consecutive_steps_in_the_same_direction():
    """
    `[("N", 3), ("N", 2)]` is not normalized into `[("N", 5)]`. Rejecting the
    unmerged form would be a purely cosmetic restriction on participants, so
    both spellings are valid and describe the same trajectory.
    """
    split = make_solution(
        cutters=[{"start": (0, 0), "steps": [("N", 3), ("N", 2), ("S", 5)]}]
    )
    merged = make_solution(cutters=[{"start": (0, 0), "steps": [("N", 5), ("S", 5)]}])
    assert split.cutters[0].steps == [("N", 3), ("N", 2), ("S", 5)]
    assert list(split.cutters[0].positions()) == list(merged.cutters[0].positions())


def test_solution_survives_a_json_round_trip():
    solution = make_solution()
    assert (
        CGSHOP2027Solution.model_validate_json(solution.model_dump_json()) == solution
    )


def test_steps_are_serialized_as_pairs_not_objects():
    """The wire format of a step is `["N", 3]`, not `{"direction": ...}`."""
    dumped = make_solution(cutters=[L_PATH]).model_dump_json()
    assert '"steps":[["N",3],["I",2],["W",1],["E",1],["S",3]]' in dumped


# --- rejected input ---


@pytest.mark.parametrize("direction", ["north", "n", "U", "", "NE"])
def test_solution_rejects_an_unknown_direction(direction: str):
    with pytest.raises(ValidationError, match="'N', 'E', 'S', 'W' or 'I'"):
        make_solution(cutters=[{"start": (0, 0), "steps": [(direction, 1)]}])


@pytest.mark.parametrize("units", [0, -1])
def test_solution_rejects_non_positive_units(units: int):
    with pytest.raises(ValidationError, match="greater than 0"):
        make_solution(cutters=[{"start": (0, 0), "steps": [("N", units)]}])


def test_solution_rejects_fractional_units():
    with pytest.raises(ValidationError, match="fractional part"):
        make_solution(cutters=[{"start": (0, 0), "steps": [("N", 1.5)]}])


def test_solution_rejects_a_step_without_a_unit_count():
    with pytest.raises(ValidationError) as exc_info:
        make_solution(cutters=[{"start": (0, 0), "steps": [("N",)]}])
    error = exc_info.value.errors()[0]
    assert error["type"] == "missing"
    assert error["loc"] == ("cutters", 0, "steps", 0, 1)


def test_solution_rejects_an_overlong_step():
    with pytest.raises(ValidationError, match="at most 2 items"):
        make_solution(cutters=[{"start": (0, 0), "steps": [("N", 1, 1)]}])


def test_solution_rejects_a_cutter_that_does_not_return_to_its_start():
    with pytest.raises(ValidationError, match="does not return to its origin"):
        make_solution(cutters=[OPEN_PATH])


def test_the_index_of_the_wandering_cutter_is_reported():
    with pytest.raises(ValidationError, match=r"Cutter #1 "):
        make_solution(cutters=[L_PATH, OPEN_PATH])


def test_a_detour_that_comes_back_is_not_a_wandering_cutter():
    """Returning home is about the end position, not about the route taken."""
    detour = {"start": (0, 0), "steps": [("N", 4), ("E", 2), ("S", 4), ("W", 2)]}
    assert make_solution(cutters=[detour]).cutters[0].end == (0, 0)


def test_solution_rejects_a_foreign_content_type():
    with pytest.raises(ValidationError, match="CGSHOP2027_Solution"):
        make_solution(content_type="CGSHOP2026_Solution")


@pytest.mark.parametrize("field", ["instance_uid", "cutters"])
def test_solution_requires_every_problem_field(field: str):
    fields = {"instance_uid": "demo", "cutters": [L_PATH]}
    del fields[field]
    with pytest.raises(ValidationError) as exc_info:
        CGSHOP2027Solution(**fields)
    error = exc_info.value.errors()[0]
    assert error["type"] == "missing"
    assert error["loc"] == (field,)


def test_cutter_path_requires_steps():
    """
    Like `PolyominoWithHoles.inner_boundaries`, `steps` is required rather than
    defaulting to an empty list, so a stationary cutter still says so explicitly.
    """
    with pytest.raises(ValidationError) as exc_info:
        make_solution(cutters=[{"start": (0, 0)}])
    error = exc_info.value.errors()[0]
    assert error["type"] == "missing"
    assert error["loc"] == ("cutters", 0, "steps")


@pytest.mark.parametrize(
    "dump_options",
    [{}, {"exclude_defaults": True}, {"exclude_unset": True}, {"exclude_none": True}],
    ids=["plain", "exclude_defaults", "exclude_unset", "exclude_none"],
)
def test_empty_step_and_cutter_lists_are_always_serialized(
    dump_options: dict[str, bool],
):
    """
    Neither an idle cutter nor an empty solution may be written out with the
    key omitted; being required fields is what guarantees this.
    """
    dumped = make_solution(cutters=[STATIONARY_PATH]).model_dump_json(**dump_options)
    assert '"steps":[]' in dumped
    assert '"cutters":[]' in make_solution(cutters=[]).model_dump_json(**dump_options)


def test_solution_reports_the_location_of_a_bad_step():
    with pytest.raises(ValidationError) as exc_info:
        make_solution(
            cutters=[
                STATIONARY_PATH,
                {"start": (0, 0), "steps": [("N", 1), ("E", 1), ("north", 1)]},
            ]
        )
    assert exc_info.value.errors()[0]["loc"] == ("cutters", 1, "steps", 2, 0)


# --- trajectory helpers ---


def test_duration_counts_idle_time():
    assert CutterPath(**OPEN_PATH).duration == 6
    assert CutterPath(**STATIONARY_PATH).duration == 0


def test_end_is_the_last_position():
    path = CutterPath(**OPEN_PATH)
    assert path.end == (2, 8)
    assert path.end == list(path.positions())[-1]


def test_positions_traces_the_trajectory_step_by_step():
    """Idling repeats the position, so time and index stay in sync."""
    assert list(CutterPath(**OPEN_PATH).positions()) == [
        (3, 5),
        (3, 6),
        (3, 7),
        (3, 8),  # arrived after three units north
        (3, 8),  # idling
        (3, 8),
        (2, 8),  # one unit west
    ]


def test_waypoints_yields_one_point_per_step_boundary():
    path = CutterPath(**OPEN_PATH)
    waypoints = list(path.waypoints())
    assert len(waypoints) == len(path.steps) + 1
    assert waypoints[0] == path.start
    assert waypoints[-1] == path.end


def test_waypoints_repeats_the_position_while_idling():
    """An idle step is a corner that does not turn, so the point repeats."""
    assert list(CutterPath(**OPEN_PATH).waypoints()) == [
        (3, 5),
        (3, 8),  # three units north
        (3, 8),  # idling
        (2, 8),  # one unit west
    ]


def test_waypoints_agree_with_positions_at_the_step_boundaries():
    """
    The two views of a trajectory must not drift: sampling `positions()` at the
    cumulative step durations has to reproduce `waypoints()` exactly.
    """
    path = CutterPath(**OPEN_PATH)
    positions = list(path.positions())
    boundaries, elapsed = [0], 0
    for _, units in path.steps:
        elapsed += units
        boundaries.append(elapsed)
    assert [positions[t] for t in boundaries] == list(path.waypoints())


def test_positions_yields_one_point_per_time_step():
    for path in (CutterPath(**OPEN_PATH), CutterPath(**STATIONARY_PATH)):
        assert len(list(path.positions())) == path.duration + 1


@pytest.mark.parametrize(
    ("direction", "expected"),
    [("N", (0, 1)), ("E", (1, 0)), ("S", (0, -1)), ("W", (-1, 0)), ("I", (0, 0))],
)
def test_each_direction_moves_the_expected_way(
    direction: str, expected: tuple[int, int]
):
    """North is +y, matching the orientation convention of `PointSequence.is_ccw`."""
    assert CutterPath(start=(0, 0), steps=[(direction, 1)]).end == expected
