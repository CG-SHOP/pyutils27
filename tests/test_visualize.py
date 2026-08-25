from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from conftest import footprint_polygon
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as PolygonPatch
from shapely.geometry import Point
from shapely.geometry import Polygon as ShapelyPolygon

from cgshop2027_pyutils.schemas import CGSHOP2027Instance, CGSHOP2027Solution
from cgshop2027_pyutils.verify import SolutionValidator
from cgshop2027_pyutils.visualize import (
    create_instance_plot,
    create_solution_animation,
    create_solution_plot,
)

UNIT_CUTTER = {"x": [0, 1, 1, 0], "y": [0, 0, 1, 1]}
# An L, so that no symmetry can hide a misplaced anchor; every center below is
# a vertex of it, which the schema accepts because the cutter is closed.
L_CUTTER = {"x": [0, 2, 2, 1, 1, 0], "y": [0, 0, 1, 1, 2, 2]}
L_CENTERS = [(0, 0), (1, 1), (2, 0), (0, 2), (1, 2)]


def cutter_patches(axes) -> list[PolygonPatch]:
    """The cutter outlines of a plot; the region is drawn as a `PathPatch`."""
    return [patch for patch in axes.patches if isinstance(patch, PolygonPatch)]


def drawn_area(patch: PolygonPatch) -> ShapelyPolygon:
    return ShapelyPolygon(patch.get_xy())


def misplaced(patch: PolygonPatch, instance, position) -> float:
    """
    How much of the drawn outline does not coincide with the cutter placed at
    `position`. Zero iff the plot anchors the cutter at its center.
    """
    expected = footprint_polygon(instance, position)
    return drawn_area(patch).symmetric_difference(expected).area


def make_instance(
    size: int = 4, holes=(), cutter=UNIT_CUTTER, center=(0, 0), cutters: int = 1
):
    return CGSHOP2027Instance(
        instance_uid="demo",
        region_to_cover={
            "outer_boundary": {"x": [0, size, size, 0], "y": [0, 0, size, size]},
            "inner_boundaries": list(holes),
        },
        cutter=cutter,
        cutter_center=center,
        number_of_cutters=cutters,
    )


def tour(*points):
    """A closed tour from its corner points, in the schema's coordinate lists."""
    return {"x": [x for x, _ in points], "y": [y for _, y in points]}


def boustrophedon(y0: int = 0, rows: int = 4, width: int = 3):
    """A closed tour sweeping `rows` rows with a unit cutter."""
    points, left = [], True
    for row in range(rows):
        y = y0 + row
        points += [(0, y), (width, y)] if left else [(width, y), (0, y)]
        left = not left
    if points[-1] != (width, y0) and points[-1][0] != 0:
        # An odd number of rows ends on the right, where the closing edge back
        # to the start would be diagonal; drop down the right side first.
        points.append((width, y0))
    return tour(*points)


FULL_SWEEP = boustrophedon()
HOLE = {"x": [1, 1, 3, 3], "y": [1, 3, 3, 1]}


def make_solution(*tours):
    return CGSHOP2027Solution(instance_uid="demo", tours=list(tours) or [FULL_SWEEP])


def renders(figure: Figure) -> bool:
    """Draws the figure to prove the artists survive a real render pass."""
    figure.savefig(BytesIO(), format="png", dpi=50)
    return True


# --------------------------------------------------------------------------
# instance plot
# --------------------------------------------------------------------------


def test_instance_plot_has_a_panel_for_the_region_and_the_cutter():
    figure = create_instance_plot(make_instance())
    assert isinstance(figure, Figure)
    assert len(figure.axes) == 2


def test_instance_plot_renders():
    assert renders(create_instance_plot(make_instance(holes=[HOLE])))


def test_instance_plot_covers_the_whole_region():
    figure = create_instance_plot(make_instance(size=7))
    region_axes = figure.axes[0]
    x_min, x_max = region_axes.get_xlim()
    y_min, y_max = region_axes.get_ylim()
    assert x_min <= 0
    assert x_max >= 7
    assert y_min <= 0
    assert y_max >= 7


def test_instance_plot_reports_the_cell_counts():
    figure = create_instance_plot(make_instance(size=4, holes=[HOLE]))
    assert "12 cells" in figure.axes[0].get_title()  # 16 minus a 2x2 hole
    assert "demo" in figure.get_suptitle()


# --------------------------------------------------------------------------
# solution plot
# --------------------------------------------------------------------------


def test_solution_plot_renders():
    assert renders(create_solution_plot(make_instance(), make_solution()))


def test_solution_plot_reports_a_feasible_solution():
    figure = create_solution_plot(make_instance(), make_solution())
    assert "feasible" in figure.get_suptitle()
    assert "0 cells uncovered" in figure.axes[0].get_title()


def test_solution_plot_reports_the_gaps_of_an_incomplete_solution():
    figure = create_solution_plot(make_instance(), make_solution(boustrophedon(rows=2)))
    assert "1 error" in figure.get_suptitle()
    assert "8 cells uncovered" in figure.axes[0].get_title()


def test_solution_plot_renders_holes_and_several_cutters():
    solution = make_solution(boustrophedon(rows=2), boustrophedon(y0=2, rows=2))
    assert renders(
        create_solution_plot(make_instance(holes=[HOLE], cutters=2), solution)
    )


def test_solution_plot_includes_area_swept_outside_the_region():
    """The view has to grow to fit a cutter that wanders off."""
    detour = tour((0, 0), (-3, 0))
    figure = create_solution_plot(make_instance(), make_solution(detour))
    x_min, _ = figure.axes[0].get_xlim()
    assert x_min <= -3


# --------------------------------------------------------------------------
# the anchor is the cutter's center
# --------------------------------------------------------------------------


@pytest.mark.parametrize("center", L_CENTERS)
def test_the_plotted_cutter_is_placed_by_its_center(center):
    """
    The outline drawn at a cutter's start has to cover exactly the area the
    cutter occupies there. The plot shifts the cutter by its center just as the
    verifier does, and if the two ever disagreed on the sign the picture would
    contradict the verdict printed above it.
    """
    instance = make_instance(size=4, cutter=L_CUTTER, center=center)
    solution = make_solution(tour((2, 2)))
    figure = create_solution_plot(instance, solution)
    (patch,) = cutter_patches(figure.axes[0])
    assert misplaced(patch, instance, (2, 2)) == pytest.approx(0.0)


@pytest.mark.parametrize("center", L_CENTERS)
def test_the_instance_plot_draws_the_cutter_around_its_anchor(center):
    """
    The cutter panel is drawn relative to the center and marks it at the
    origin, so the marker must lie on the outline rather than off to one side.
    """
    instance = make_instance(cutter=L_CUTTER, center=center)
    figure = create_instance_plot(instance)
    (patch,) = cutter_patches(figure.axes[1])
    outline = drawn_area(patch)
    assert outline.covers(Point(0, 0))
    assert outline.area == pytest.approx(len(SolutionValidator(instance).cutter))


# --------------------------------------------------------------------------
# animation
# --------------------------------------------------------------------------


def frames_of(animation: FuncAnimation) -> list[int]:
    return list(animation.new_frame_seq())


def covered_cells(image) -> int:
    """
    The number of painted cells of a coverage raster.

    Goes through `getmaskarray` on purpose: once every cell is covered numpy
    collapses the all-False mask to the `nomask` scalar, and `~image.mask`
    would then count 1 instead of the whole grid.
    """
    return int((~np.ma.getmaskarray(image.get_array())).sum())


def test_animation_shows_every_step_of_a_short_tour():
    solution = make_solution()
    animation = create_solution_animation(make_instance(), solution)
    assert isinstance(animation, FuncAnimation)
    assert len(frames_of(animation)) == solution.max_tour_length + 1


def test_an_explicit_step_is_honoured():
    animation = create_solution_animation(make_instance(), make_solution(), step=4)
    length = make_solution().max_tour_length
    assert len(frames_of(animation)) == len(range(0, length + 1, 4)) + 1


def test_the_frame_budget_is_respected():
    long_tour = tour((0, 0), (200, 0))
    animation = create_solution_animation(
        make_instance(), make_solution(long_tour), max_frames=10
    )
    # The final step is always shown, which may cost one frame over budget.
    assert len(frames_of(animation)) <= 11


def test_the_last_frame_is_the_final_step():
    solution = make_solution(tour((0, 0), (7, 0)))
    animation = create_solution_animation(make_instance(), solution, max_frames=4)
    for frame in frames_of(animation):
        animation._func(frame)
    assert "step 14 / 14" in animation._fig.axes[0].get_title()


def test_a_stationary_solution_animates_a_single_frame():
    animation = create_solution_animation(make_instance(), make_solution(tour((0, 0))))
    assert len(frames_of(animation)) == 1


def test_a_cutter_with_a_short_tour_waits_at_its_start():
    """
    Every cutter travels at the same speed, so a short tour is done early; its
    cutter then stays parked where it started.
    """
    parked = tour((6, 6))
    solution = make_solution(FULL_SWEEP, parked)
    animation = create_solution_animation(make_instance(cutters=2), solution)
    instance = make_instance(cutters=2)
    for frame in frames_of(animation):
        (_, _, patch) = animation._func(frame)
        assert misplaced(patch, instance, (6, 6)) == pytest.approx(0.0)


@pytest.mark.parametrize("max_frames", [300, 5, 2])
def test_coverage_grows_monotonically_to_the_full_swept_area(max_frames: int):
    """
    Striding must not lose the placements between two frames: whatever the
    frame budget, the animation has to end on exactly the area the verifier
    reports as swept.
    """
    instance, solution = make_instance(), make_solution()
    expected = len(SolutionValidator(instance).swept_cells(solution))
    animation = create_solution_animation(instance, solution, max_frames=max_frames)
    covered = []
    for frame in frames_of(animation):
        (image, *_) = animation._func(frame)
        covered.append(covered_cells(image))
    assert covered == sorted(covered)
    assert covered[-1] == expected
    assert covered[0] < covered[-1]


@pytest.mark.parametrize("center", L_CENTERS)
def test_the_animated_cutter_and_its_trail_are_anchored_at_the_center(center):
    """
    Both halves of a frame are placed by the center -- the moving outline and
    the coverage stamped behind it -- so both are checked against the verifier
    for a cutter whose center is not its corner.
    """
    instance = make_instance(size=4, cutter=L_CUTTER, center=center)
    solution = make_solution()
    animation = create_solution_animation(instance, solution)

    (_, patch) = animation._func(0)
    assert misplaced(patch, instance, solution.tours[0].start) == pytest.approx(0.0)

    for frame in frames_of(animation):
        (image, *_) = animation._func(frame)
    assert covered_cells(image) == len(
        SolutionValidator(instance).swept_cells(solution)
    )


def test_animation_restarts_cleanly_when_it_loops():
    """Replaying from the first frame must not keep the coverage of the last."""
    instance, solution = make_instance(), make_solution()
    animation = create_solution_animation(instance, solution)
    frames = frames_of(animation)
    for frame in frames:
        animation._func(frame)
    (image, *_) = animation._func(frames[0])
    first_again = covered_cells(image)
    assert first_again == len(SolutionValidator(instance).cutter)


def test_animation_saves_as_a_gif(tmp_path: Path):
    path = tmp_path / "tour.gif"
    animation = create_solution_animation(
        make_instance(), make_solution(), max_frames=4
    )
    animation.save(path, writer="pillow", dpi=40)
    assert path.stat().st_size > 0


# --------------------------------------------------------------------------
# bare figures
# --------------------------------------------------------------------------


def undecorated(figure: Figure) -> bool:
    """Nothing but the drawing: no axes, no grid, no titles anywhere."""
    return (
        not figure.get_suptitle()
        and all(not axes.axison for axes in figure.axes)
        and all(not axes.get_title() for axes in figure.axes)
    )


def test_a_bare_solution_plot_drops_the_frame_around_the_drawing():
    figure = create_solution_plot(make_instance(), make_solution(), bare=True)
    assert undecorated(figure)
    assert renders(figure)


def test_a_bare_solution_plot_keeps_everything_that_is_the_drawing():
    """
    The point is to remove the frame, not the picture: the coverage rasters,
    the region, the tours and the cutter outlines all have to survive.
    """
    instance, solution = make_instance(), make_solution(boustrophedon(rows=2))
    plain = create_solution_plot(instance, solution)
    bare = create_solution_plot(instance, solution, bare=True)
    for attribute in ("images", "lines", "patches"):
        assert len(getattr(bare.axes[0], attribute)) == len(
            getattr(plain.axes[0], attribute)
        )


def test_a_bare_instance_plot_drops_the_frame_around_both_panels():
    figure = create_instance_plot(make_instance(), bare=True)
    assert undecorated(figure)
    assert renders(figure)


def test_a_bare_animation_never_labels_a_frame():
    """The step counter is part of the frame, so it goes with it."""
    animation = create_solution_animation(
        make_instance(), make_solution(), max_frames=4, bare=True
    )
    for frame in frames_of(animation):
        animation._func(frame)
        assert undecorated(animation._fig)


def test_a_bare_figure_is_trimmed_to_the_shape_of_the_drawing():
    """
    A square figure around a drawing three times as wide as it is tall is
    mostly margin, and margin is exactly what `bare` is for.
    """
    solution = make_solution(FULL_SWEEP, tour((0, 0), (30, 0)))
    instance = make_instance(cutters=2)
    plain = create_solution_plot(instance, solution)
    bare = create_solution_plot(instance, solution, bare=True)
    assert plain.get_size_inches()[0] == pytest.approx(plain.get_size_inches()[1])
    width, height = bare.get_size_inches()
    axes = bare.axes[0]
    (x_min, x_max), (y_min, y_max) = axes.get_xlim(), axes.get_ylim()
    assert width / height == pytest.approx((x_max - x_min) / (y_max - y_min))
