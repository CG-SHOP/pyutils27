from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from conftest import footprint_polygon
from matplotlib import colormaps
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as PolygonPatch
from shapely.geometry import Point
from shapely.geometry import Polygon as ShapelyPolygon

from cgshop2027_pyutils.schemas import CGSHOP2027Instance, CGSHOP2027Solution
from cgshop2027_pyutils.verify import SolutionValidator
from cgshop2027_pyutils.visualize import (
    _visit_counts,
    create_instance_plot,
    create_solution_animation,
    create_solution_plot,
)

UNIT_CUTTER = {"x": [0, 1, 1, 0], "y": [0, 0, 1, 1]}
# Wide enough that neighbouring placements overlap, so a plain tour already
# passes over a cell more than once.
BLOCK_CUTTER = {"x": [0, 2, 2, 0], "y": [0, 0, 2, 2]}
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
# coverage is coloured per cutter and shaded by how often it passed
# --------------------------------------------------------------------------


def coverage_rasters(axes) -> list:
    """The coverage layers of a plot, in the order they were drawn."""
    return list(axes.images)


def cutter_colors(count: int) -> list[tuple[float, float, float]]:
    """The colour the plots give each cutter, in tour order."""
    palette = colormaps["tab10"]
    return [tuple(palette(index % palette.N)[:3]) for index in range(count)]


def shade(image, count: int) -> tuple[float, ...]:
    """The colour a raster paints a cell that was covered `count` times."""
    return tuple(image.get_cmap()(image.norm(count))[:3])


def luminance(color) -> float:
    return 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]


TWO_CUTTERS = (boustrophedon(rows=2), boustrophedon(y0=2, rows=2))


def two_cutter_plot():
    return create_solution_plot(make_instance(cutters=2), make_solution(*TWO_CUTTERS))


def test_each_cutter_gets_its_own_coverage_layer():
    axes = two_cutter_plot().axes[0]
    # One coverage layer per cutter, plus the uncovered-cell layer.
    assert len(coverage_rasters(axes)) == 3


def test_a_cutter_shades_its_coverage_in_its_own_colour():
    """
    The fill has to name the cutter that swept it: washed out towards white
    where it passed once, deepened towards black where it passed most, but the
    same hue as the tour drawn over it throughout.
    """
    axes = two_cutter_plot().axes[0]
    for raster, own in zip(coverage_rasters(axes)[:2], cutter_colors(2), strict=True):
        light, dark = shade(raster, 1), shade(raster, raster.norm.vmax)
        assert luminance(light) > luminance(own) > luminance(dark)
        # Both ends sit on the line through the cutter's colour, so the hue --
        # which channel leads -- never changes.
        assert np.argmax(light) == np.argmax(own) == np.argmax(dark)


def test_more_passes_are_drawn_darker():
    (raster, *_) = coverage_rasters(two_cutter_plot().axes[0])
    shades = [luminance(shade(raster, n)) for n in range(1, int(raster.norm.vmax) + 1)]
    assert shades == sorted(shades, reverse=True)


def test_two_cutters_are_told_apart_by_colour():
    first, second = coverage_rasters(two_cutter_plot().axes[0])[:2]
    assert shade(first, 1) != shade(second, 1)


def test_every_cutter_is_measured_against_the_same_scale():
    """
    A shade has to mean the same number of passes whichever cutter drew it,
    which it only does if they share one scale rather than each stretching
    its own coverage over the whole ramp.
    """
    instance = make_instance(size=6, cutter=BLOCK_CUTTER, cutters=2)
    solution = make_solution(
        boustrophedon(rows=3),
        tour((0, 2), (4, 2), (4, 4), (2, 4), (2, 0), (0, 0)),
    )
    grids = _visit_counts(SolutionValidator(instance), solution)
    # Without this the test would hold for a per-cutter scale just as well.
    assert grids[0].maximum != grids[1].maximum
    rasters = coverage_rasters(create_solution_plot(instance, solution).axes[0])
    assert len({(r.norm.vmin, r.norm.vmax) for r in rasters[:2]}) == 1


def test_a_cell_swept_once_is_shaded_the_same_whatever_the_busiest_cell():
    """
    The light end stands for a single pass, not for `1 / busiest`, so a picture
    of a tidy solution is not washed out by a wasteful one.
    """
    tidy = create_solution_plot(make_instance(), make_solution())
    # A tour that keeps going back over the same row raises the busiest count.
    wasteful = create_solution_plot(
        make_instance(), make_solution(boustrophedon(rows=4, width=1))
    )
    (a, *_), (b, *_) = (coverage_rasters(figure.axes[0]) for figure in (tidy, wasteful))
    assert shade(a, 1) == pytest.approx(shade(b, 1))


def test_the_scale_spans_at_least_two_passes():
    """With nothing covered twice there is no range to grade, and no crash."""
    (raster, *_) = coverage_rasters(
        create_solution_plot(
            make_instance(size=1), make_solution(tour((0, 0), (1, 0)))
        ).axes[0]
    )
    assert raster.norm.vmax >= 2


# --------------------------------------------------------------------------
# how often each cutter passed over a cell
# --------------------------------------------------------------------------


def test_a_cutter_that_comes_back_counts_again():
    """
    A tour that crosses itself passes over the crossing twice, and the cell
    there has to say 2 while the rest of the tour says 1.
    """
    instance = make_instance(size=5)
    # A bowtie whose two strokes both run through (2, 2).
    crossing = make_solution(tour((0, 2), (4, 2), (4, 4), (2, 4), (2, 0), (0, 0)))
    (grid,) = _visit_counts(SolutionValidator(instance), crossing)
    assert grid.maximum == 2
    at_crossing = grid.counts[2 - grid.origin[1], 2 - grid.origin[0]]
    assert at_crossing == 2
    elsewhere = grid.counts[2 - grid.origin[1], 1 - grid.origin[0]]
    assert elsewhere == 1


def test_the_lap_counts_its_start_once():
    """
    A tour closes onto its own start, and the cutter is there once, not twice.
    Every cell is then reached by exactly `length` placements of the cutter.
    """
    instance = make_instance(size=6)
    solution = make_solution(tour((0, 0), (4, 0), (4, 4), (0, 4)))
    (grid,) = _visit_counts(SolutionValidator(instance), solution)
    expected = solution.tours[0].length * len(SolutionValidator(instance).cutter)
    assert int(grid.counts.sum()) == expected


def test_a_stationary_cutter_counts_once():
    instance = make_instance()
    (grid,) = _visit_counts(SolutionValidator(instance), make_solution(tour((2, 2))))
    assert grid.maximum == 1
    assert int(grid.counts.sum()) == len(SolutionValidator(instance).cutter)


def test_counts_cover_exactly_what_the_verifier_calls_swept():
    """
    Splitting the coverage per cutter and counting it must neither lose nor
    invent a cell, or the picture would contradict the verdict above it.
    """
    instance = make_instance(cutters=2)
    solution = make_solution(*TWO_CUTTERS)
    validator = SolutionValidator(instance)
    grids = _visit_counts(validator, solution)
    for index, grid in enumerate(grids):
        alone = solution.model_copy(update={"tours": [solution.tours[index]]})
        assert set(grid.nonzero().cells()) == set(validator.swept_cells(alone).cells())
    union = set().union(*(set(grid.nonzero().cells()) for grid in grids))
    assert union == set(validator.swept_cells(solution).cells())


def test_placements_outside_the_regions_reach_are_left_out():
    """
    The verifier clips the anchors to where the cutter can still touch the
    region; the counts have to drop the same placements, or the drawn area
    would run past the one the verdict is based on.
    """
    instance = make_instance(size=4)
    detour = make_solution(tour((0, 0), (-30, 0), (-30, 1), (0, 1)))
    validator = SolutionValidator(instance)
    (grid,) = _visit_counts(validator, detour)
    assert set(grid.nonzero().cells()) == set(validator.swept_cells(detour).cells())


# --------------------------------------------------------------------------
# the animation counts the same way the plot does
# --------------------------------------------------------------------------


def test_the_animation_paints_one_raster_per_cutter():
    animation = create_solution_animation(
        make_instance(cutters=2), make_solution(*TWO_CUTTERS)
    )
    rasters = coverage_rasters(animation._fig.axes[0])
    assert len(rasters) == 2
    assert shade(rasters[0], 1) != shade(rasters[1], 1)


@pytest.mark.parametrize("max_frames", [300, 5, 2])
@pytest.mark.parametrize("wanders", [False, True])
def test_the_animation_ends_on_exactly_the_counts_the_plot_draws(
    max_frames: int, *, wanders: bool
):
    """
    Whatever the stride, the last frame has to leave each cutter's counts as
    `create_solution_plot` would draw them -- padding for a cutter that finished
    early included, since standing still is not another pass, and a cutter that
    wanders out of the region's reach, whose placements out there the plot drops.
    """
    # The cutter has to be wider than a cell: a single-cell one placed out of
    # the region's reach paints only out there, and the clipping would not show.
    instance = make_instance(size=6, cutter=BLOCK_CUTTER, cutters=2)
    # One short tour and one long one, so a cutter is parked while the other runs.
    short = (
        tour((0, 0), (-30, 0), (-30, 1), (0, 1)) if wanders else tour((0, 0), (1, 0))
    )
    solution = make_solution(short, boustrophedon(rows=4))
    animation = create_solution_animation(instance, solution, max_frames=max_frames)
    for frame in frames_of(animation):
        artists = animation._func(frame)
    rasters = rasters_of(animation, artists)
    swept = SolutionValidator(instance).swept_cells(solution)
    for raster, grid in zip(
        rasters, _visit_counts(SolutionValidator(instance), solution), strict=True
    ):
        expected = np.zeros_like(passes(raster))
        dx = grid.origin[0] - swept.origin[0]
        dy = grid.origin[1] - swept.origin[1]
        expected[dy : dy + grid.counts.shape[0], dx : dx + grid.counts.shape[1]] = (
            grid.counts
        )
        assert np.array_equal(passes(raster), expected)


def test_the_animated_scale_does_not_shift_as_coverage_grows():
    """
    The scale is fixed from the finished counts, so a cell keeps its shade
    instead of being repainted every time some other cell overtakes it.
    """
    animation = create_solution_animation(
        make_instance(cutters=2), make_solution(*TWO_CUTTERS)
    )
    rasters = coverage_rasters(animation._fig.axes[0])
    before = [(r.norm.vmin, r.norm.vmax) for r in rasters]
    for frame in frames_of(animation):
        animation._func(frame)
    assert [(r.norm.vmin, r.norm.vmax) for r in rasters] == before


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


def painted(images) -> int:
    """The cells some cutter has painted, counting a shared cell once."""
    masks = [~np.ma.getmaskarray(image.get_array()) for image in images]
    return int(np.logical_or.reduce(masks).sum())


def passes(image) -> np.ndarray:
    """The pass count behind each cell of a coverage raster, 0 where unpainted."""
    return np.ma.filled(image.get_array(), 0).astype(np.int64)


def rasters_of(animation, artists) -> list:
    """The coverage rasters among the artists a frame returned."""
    return list(artists[: len(animation._fig.axes[0].images)])


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
        (*_, patch) = animation._func(frame)
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
        artists = animation._func(frame)
        covered.append(painted(rasters_of(animation, artists)))
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
        artists = animation._func(frame)
    assert painted(rasters_of(animation, artists)) == len(
        SolutionValidator(instance).swept_cells(solution)
    )


def test_animation_restarts_cleanly_when_it_loops():
    """Replaying from the first frame must not keep the coverage of the last."""
    instance, solution = make_instance(), make_solution()
    animation = create_solution_animation(instance, solution)
    frames = frames_of(animation)
    for frame in frames:
        animation._func(frame)
    artists = animation._func(frames[0])
    first_again = painted(rasters_of(animation, artists))
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
