"""
Matplotlib plots for CG:SHOP 2027 instances and solutions.

Everything here returns a matplotlib object for the caller to show, save or
embed; nothing is drawn to screen and no backend is chosen.

Usage:
- `create_instance_plot(instance)` for the region and the cutter,
- `create_solution_plot(instance, solution)` for a solution, with the swept
  area and any gaps marked,
- `create_solution_animation(instance, solution)` for the cutters moving along
  their tours, filling in coverage as they go.

Every one of them takes `bare=True`, which strips the figure down to the
drawing itself -- no axes, no grid, no titles -- for a picture that goes into a
paper or a slide rather than in front of someone debugging.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.colors import (
    Colormap,
    LinearSegmentedColormap,
    ListedColormap,
    Normalize,
    to_rgb,
)
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.patches import PathPatch
from matplotlib.patches import Polygon as PolygonPatch
from matplotlib.path import Path as MplPath

from .grid import CellCounts, CellSet, _blit_counts, dilate_counts
from .schemas import CGSHOP2027Instance, CGSHOP2027Solution, CutterTour
from .schemas.instance import PointSequence, PolyominoWithHoles
from .verify import SolutionValidator

__all__ = [
    "create_instance_plot",
    "create_solution_animation",
    "create_solution_plot",
]

# Anything matplotlib accepts as a colour; here a hex string or an RGB(A) tuple.
_Color = str | tuple[float, ...]

_REGION_FACE = "#d9d9d9"
_REGION_EDGE = "#404040"
_UNCOVERED = "#d62728"
_CUTTER = "#1f77b4"
# The two ends of a cutter's coverage ramp, as distances from its own colour:
# washed out towards white where it passed once, pushed towards black where it
# passed most. The span has to be wide enough that one more pass is a visible
# step, which is what the shading is for.
_SWEPT_LIGHTEN = 0.85
_SWEPT_DARKEN = 0.6
# Opaque enough to hold that span, but short of hiding what is underneath: the
# cutters are drawn one over another, so where two of them sweep the same cell
# the one below has to show through the one above. The ramp is deepened to pay
# for what the transparency takes back off the span.
_SWEPT_ALPHA = 0.7


def _ring_vertices(ring: PointSequence) -> list[tuple[int, int]]:
    return list(zip(ring.x, ring.y, strict=True))


def _polyomino_path(polyomino: PolyominoWithHoles) -> MplPath:
    """
    The polyomino as a single compound path, one closed sub-path per boundary.

    The holes need no special treatment: under the nonzero winding rule they
    are cut out precisely because the schema requires the outer boundary to run
    counter-clockwise and the inner ones clockwise.
    """
    vertices: list[tuple[int, int]] = []
    codes: list[int] = []
    for ring in (polyomino.outer_boundary, *polyomino.inner_boundaries):
        points = _ring_vertices(ring)
        vertices.extend([*points, points[0]])
        codes.extend(
            [MplPath.MOVETO, *[MplPath.LINETO] * (len(points) - 1), MplPath.CLOSEPOLY]
        )
    return MplPath(vertices, codes)


def _lighten(color: _Color, amount: float) -> tuple[float, ...]:
    """
    The colour blended `amount` of the way towards white.

    :param amount: 0 leaves the colour as it is, 1 turns it white.
    """
    return tuple(channel + (1.0 - channel) * amount for channel in to_rgb(color))


def _darken(color: _Color, amount: float) -> tuple[float, ...]:
    """
    The colour blended `amount` of the way towards black.

    :param amount: 0 leaves the colour as it is, 1 turns it black.
    """
    return tuple(channel * (1.0 - amount) for channel in to_rgb(color))


def _shading(color: _Color) -> LinearSegmentedColormap:
    """
    The ramp a cutter's coverage is drawn with: washed out where the cutter
    passed the fewest times, deepened past its own colour where it passed the
    most. It stays recognizably that cutter's colour throughout, since both
    ends only move it along its own line to white and to black.
    """
    return LinearSegmentedColormap.from_list(
        "coverage",
        [_lighten(color, _SWEPT_LIGHTEN), _darken(color, _SWEPT_DARKEN)],
    )


def _draw_cells(
    ax: Axes, cells: CellSet, color: _Color, alpha: float = 1.0, zorder: float = 1.0
) -> AxesImage:
    """
    Draws a cell set as a raster.

    A cell set already *is* a raster, so `imshow` renders it in one artist no
    matter how many cells it holds; a collection of unit squares would not
    survive a region of a million cells.
    """
    x_min, y_min, x_max, y_max = cells.bounds
    return ax.imshow(
        _as_image(cells.mask),
        origin="lower",
        interpolation="nearest",
        extent=(x_min, x_max + 1, y_min, y_max + 1),
        cmap=ListedColormap([color]),
        vmin=0,
        vmax=1,
        alpha=alpha,
        zorder=zorder,
    )


def _draw_counts(
    ax: Axes,
    visits: CellCounts,
    color: _Color,
    busiest: int,
    *,
    alpha: float = 1.0,
    zorder: float = 1.0,
) -> AxesImage:
    """
    Draws a count grid as a raster shaded by how often each cell was covered.

    `busiest` sets the dark end of the scale for every cutter alike, so the
    same shade means the same number of passes whoever drew it.
    """
    x_min, y_min, x_max, y_max = visits.bounds
    return ax.imshow(
        _as_count_image(visits.counts),
        origin="lower",
        interpolation="nearest",
        extent=(x_min, x_max + 1, y_min, y_max + 1),
        cmap=_shading(color),
        norm=Normalize(vmin=1, vmax=busiest),
        alpha=alpha,
        zorder=zorder,
    )


def _as_image(mask: np.ndarray) -> np.ma.MaskedArray:
    """Masks out the empty cells so that only the set ones are painted."""
    return np.ma.masked_where(~mask, np.ones_like(mask, dtype=np.uint8))


def _as_count_image(counts: np.ndarray) -> np.ma.MaskedArray:
    """Masks out the cells nobody ever covered, leaving the counts to shade."""
    return np.ma.masked_where(counts == 0, counts)


def _style_axes(ax: Axes, title: str | None = None, *, bare: bool = False) -> None:
    """
    The frame around a drawing: equal aspect always, and everything that is not
    the drawing itself unless `bare`.
    """
    ax.set_aspect("equal", adjustable="box")
    if bare:
        ax.set_axis_off()
        return
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.tick_params(labelsize=6)
    ax.set_xlabel("x", fontsize=7)
    ax.set_ylabel("y", fontsize=7)
    if title:
        ax.set_title(title, fontsize=9, pad=4)


def _finish(fig: Figure, title: str, *, bare: bool, ax: Axes | None = None) -> None:
    """
    Titles the figure and packs it, or -- when `bare` -- packs it edge to edge
    and, given the one axes it holds, trims the figure to the drawing's own
    aspect ratio, so that nothing is left but the picture.
    """
    if not bare:
        fig.suptitle(title, fontsize=11)
        fig.tight_layout()
        return
    if ax is not None:
        _fit_to_axes(fig, ax)
    fig.tight_layout(pad=0)


def _fit_to_axes(fig: Figure, ax: Axes) -> None:
    """Shrinks the figure to the shape of the axes' data, keeping its larger side."""
    (x_min, x_max), (y_min, y_max) = ax.get_xlim(), ax.get_ylim()
    width, height = max(x_max - x_min, 1e-9), max(y_max - y_min, 1e-9)
    inches_x, inches_y = fig.get_size_inches()
    scale = min(inches_x / width, inches_y / height)
    fig.set_size_inches(width * scale, height * scale)


def _draw_region_outline(ax: Axes, instance: CGSHOP2027Instance) -> None:
    """The region's boundary over the coverage, so gaps stay readable."""
    ax.add_patch(
        PathPatch(
            _polyomino_path(instance.region_to_cover),
            facecolor="none",
            edgecolor=_REGION_EDGE,
            linewidth=1.4,
            zorder=3,
        )
    )


def _draw_tours(
    ax: Axes,
    instance: CGSHOP2027Instance,
    solution: CGSHOP2027Solution,
    palette: Colormap,
    *,
    moving: bool,
) -> list[PolygonPatch]:
    """
    Draws each tour in its cutter's colour with the cutter sitting at its
    start, and hands the outlines back in tour order.

    :param moving: Style the outlines for an animation, where they are
                   filled and travel, rather than for a plot, where they
                   are dashed and stay put.
    """
    outline = _cutter_outline(instance)
    patches = []
    for index, tour in enumerate(solution.tours):
        color = palette(index % palette.N)
        ring = _closed(tour)
        ax.plot(
            [x for x, _ in ring],
            [y for _, y in ring],
            color=color,
            linewidth=1.0 if moving else 1.2,
            alpha=0.45 if moving else 1.0,
            zorder=2 if moving else 4,
        )
        if not moving:
            ax.plot(
                [tour.start[0]],
                [tour.start[1]],
                marker="o",
                color=color,
                markersize=5,
                zorder=5,
            )
        patch = PolygonPatch(
            _placed(outline, tour.start),
            closed=True,
            facecolor=color if moving else "none",
            edgecolor=_REGION_EDGE if moving else color,
            linestyle="-" if moving else "--",
            alpha=0.75 if moving else 1.0,
            linewidth=1.0,
            zorder=4 if moving else 5,
        )
        ax.add_patch(patch)
        patches.append(patch)
    return patches


def _set_limits(
    ax: Axes, bounds: tuple[float, float, float, float], pad_ratio: float = 0.05
) -> None:
    x_min, y_min, x_max, y_max = bounds
    pad = pad_ratio * max(x_max - x_min, y_max - y_min, 1)
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)


def _region_bounds(polyomino: PolyominoWithHoles) -> tuple[int, int, int, int]:
    outer = polyomino.outer_boundary
    return (min(outer.x), min(outer.y), max(outer.x), max(outer.y))


def _drawn_bounds(cells: CellSet) -> tuple[int, int, int, int]:
    """The coordinate box a cell set occupies, cells being unit squares."""
    x_min, y_min, x_max, y_max = cells.bounds
    return (x_min, y_min, x_max + 1, y_max + 1)


def _tour_bounds(solution: CGSHOP2027Solution) -> tuple[int, int, int, int]:
    """
    The box the anchors travel through.

    Worth including in the view of its own accord: the verifier clips the swept
    area to the region's neighbourhood, since that is all coverage depends on,
    so an excursion further out shows up only in the tour.
    """
    points = [point for tour in solution.tours for point in tour.points()]
    if not points:
        return (0, 0, 0, 0)
    return (
        min(x for x, _ in points),
        min(y for _, y in points),
        max(x for x, _ in points),
        max(y for _, y in points),
    )


def _union_bounds(*boxes: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _cutter_outline(instance: CGSHOP2027Instance) -> list[tuple[int, int]]:
    """The cutter's vertices relative to its center, i.e. to its anchor."""
    center_x, center_y = instance.cutter_center
    return [(x - center_x, y - center_y) for x, y in _ring_vertices(instance.cutter)]


def _placed(outline: list[tuple[int, int]], position) -> list[tuple[int, int]]:
    return [(x + position[0], y + position[1]) for x, y in outline]


def _closed(tour: CutterTour) -> list[tuple[int, int]]:
    """The tour's points with the first one repeated, so that it draws closed."""
    points = list(tour.points())
    return [*points, points[0]]


def _walk(tour: CutterTour, total: int) -> np.ndarray:
    """
    The center's position after every unit of travel, padded to `total` units.

    A cutter with a shorter tour is back at its start before the animation
    ends and simply stays parked there.
    """
    positions = np.array(list(tour.positions()), dtype=np.int64)
    missing = total + 1 - len(positions)
    if missing > 0:
        positions = np.concatenate([positions, np.repeat(positions[-1:], missing, 0)])
    return positions


def _visit_counts(
    validator: SolutionValidator, solution: CGSHOP2027Solution
) -> list[CellCounts]:
    """
    How often each cutter covers each cell, one count grid per tour.

    A cutter that comes back over a cell later in its tour counts again, so
    the grid says how much of the effort went where rather than merely where
    it went.

    Every grid is clipped to the same neighbourhood of the region that
    `SolutionValidator.swept_cells` clips their union to, so `nonzero()` is
    exactly that cutter's share of the swept area and the picture cannot
    drift from the verdict.
    """
    box = validator.reachable_anchors(solution)
    return [
        dilate_counts(_anchor_visits(tour, box), validator.cutter)
        for tour in solution.tours
    ]


def _anchor_visits(tour: CutterTour, box: CellSet) -> CellCounts:
    """
    How often the cutter's anchor sits on each lattice point of `box` over
    one lap of the tour.

    Each edge brings its start but not its end, so the point two consecutive
    edges share is counted once and the lap adds up to the tour's length. A
    tour with no edges is a cutter standing still, and counts once where it
    stands.
    """
    counts = np.zeros(box.mask.shape, dtype=np.int64)
    edges = list(tour.edges())
    if not edges:
        _count_run(counts, box.origin, tour.start, tour.start)
    for start, end in edges:
        step_x = (end[0] > start[0]) - (end[0] < start[0])
        step_y = (end[1] > start[1]) - (end[1] < start[1])
        _count_run(counts, box.origin, start, (end[0] - step_x, end[1] - step_y))
    return CellCounts(counts, box.origin)


def _count_run(
    counts: np.ndarray,
    origin: tuple[int, int],
    start: tuple[int, int],
    end: tuple[int, int],
) -> None:
    """
    Adds one to every lattice point of an axis-parallel run, both ends
    included, clipping whatever falls outside the grid.
    """
    (x0, y0), (x1, y1) = start, end
    origin_x, origin_y = origin
    height, width = counts.shape
    if y0 == y1:
        row = y0 - origin_y
        if not 0 <= row < height:
            return
        low, high = sorted((x0, x1))
        low, high = max(low - origin_x, 0), min(high - origin_x, width - 1)
        if low <= high:
            counts[row, low : high + 1] += 1
        return
    column = x0 - origin_x
    if not 0 <= column < width:
        return
    low, high = sorted((y0, y1))
    low, high = max(low - origin_y, 0), min(high - origin_y, height - 1)
    if low <= high:
        counts[low : high + 1, column] += 1


def _scale_max(visits: list[CellCounts]) -> int:
    """
    The count the dark end of the shading stands for.

    Two is the floor: with no cell passed over twice there is nothing to
    grade, and a cell passed over once then keeps the shade it has in every
    other picture.
    """
    return max(max((grid.maximum for grid in visits), default=1), 2)


def create_instance_plot(instance: CGSHOP2027Instance, *, bare: bool = False) -> Figure:
    """
    Plots the region to cover next to the cutter shape.

    :param bare: Draw the region and the cutter alone, without axes, grid or
                 titles.
    """
    validator = SolutionValidator(instance)
    fig, (ax_region, ax_cutter) = plt.subplots(
        1, 2, figsize=(12, 6), width_ratios=[3, 1]
    )

    ax_region.add_patch(
        PathPatch(
            _polyomino_path(instance.region_to_cover),
            facecolor=_REGION_FACE,
            edgecolor=_REGION_EDGE,
            linewidth=1.0,
        )
    )
    _set_limits(ax_region, _region_bounds(instance.region_to_cover))
    _style_axes(
        ax_region, f"Region to cover ({len(validator.region)} cells)", bare=bare
    )

    outline = _cutter_outline(instance)
    ax_cutter.add_patch(
        PolygonPatch(
            outline,
            closed=True,
            facecolor=_CUTTER,
            edgecolor=_REGION_EDGE,
            alpha=0.6,
            linewidth=1.0,
        )
    )
    ax_cutter.plot(
        [0], [0], marker="x", color=_UNCOVERED, markersize=9, markeredgewidth=2
    )
    _set_limits(
        ax_cutter,
        (
            min(x for x, _ in outline),
            min(y for _, y in outline),
            max(x for x, _ in outline),
            max(y for _, y in outline),
        ),
        pad_ratio=0.15,
    )
    _style_axes(
        ax_cutter,
        f"Cutter ({len(validator.cutter)} cells) ({instance.number_of_cutters} available)",
        bare=bare,
    )

    _finish(fig, f"CG:SHOP 2027 instance: {instance.instance_uid}", bare=bare)
    return fig


def create_solution_plot(
    instance: CGSHOP2027Instance,
    solution: CGSHOP2027Solution,
    *,
    bare: bool = False,
) -> Figure:
    """
    Plots a solution: the area the cutters sweep, the gaps they leave, and the
    tour of each cutter.

    Each cutter fills the area it sweeps in its own colour, shaded by how
    often it passed over each cell: nearly white where it went by once, the
    full colour over the cell it worked hardest. The scale is shared by all
    cutters, and where two of them sweep the same cell their fills overlap.

    :param bare: Draw the region, the coverage and the tours alone, without
                 axes, grid or titles -- the verdict included, so a gap then
                 shows only as the red it is drawn in.
    """
    validator = SolutionValidator(instance)
    errors = validator.check_for_errors(solution)
    swept = validator.swept_cells(solution)
    visits = _visit_counts(validator, solution)
    busiest = _scale_max(visits)
    uncovered = validator.uncovered_cells(solution)
    palette = colormaps["tab10"]

    fig, ax = plt.subplots(figsize=(9, 9))
    # The swept area is not clipped to the region, so effort spent just outside
    # it is visible too. It is clipped to the region's neighbourhood by the
    # verifier though, so a longer excursion shows only as a trajectory.
    for index, counted in enumerate(visits):
        _draw_counts(
            ax,
            counted,
            palette(index % palette.N),
            busiest,
            alpha=_SWEPT_ALPHA,
            zorder=1,
        )
    _draw_cells(ax, uncovered, _UNCOVERED, alpha=0.85, zorder=2)
    _draw_region_outline(ax, instance)
    _draw_tours(ax, instance, solution, palette, moving=False)

    _set_limits(
        ax,
        _union_bounds(
            _region_bounds(instance.region_to_cover),
            _drawn_bounds(swept),
            _tour_bounds(solution),
        ),
    )
    verdict = (
        "feasible"
        if not errors
        else f"{len(errors)} error{'s' if len(errors) > 1 else ''}"
    )
    _style_axes(
        ax,
        f"{len(solution.tours)} tours, longest {solution.max_tour_length} ("
        f"{len(uncovered)} cells uncovered)",
        bare=bare,
    )
    _finish(
        fig,
        f"CG:SHOP 2027 solution for {instance.instance_uid}: {verdict}",
        bare=bare,
        ax=ax,
    )
    return fig


def create_solution_animation(
    instance: CGSHOP2027Instance,
    solution: CGSHOP2027Solution,
    step: int | None = None,
    max_frames: int = 300,
    interval: int = 120,
    *,
    bare: bool = False,
) -> FuncAnimation:
    """
    Animates the cutters moving along their tours, filling in the area they
    have swept so far.

    Each cutter fills in the area it has swept in its own colour, deepening a
    cell every time it comes back over it, so both who covered what and how
    hard they worked at it stay readable as the coverage grows.

    All cutters travel at the same speed, one unit per step, so the animation
    runs for as many steps as the longest tour is long; a cutter that is
    already back where it started simply stays there.

    :param step: Units of travel between frames. Derived from `max_frames` if
                 omitted.
    :param max_frames: Frame budget used to pick `step`; the animation may end
                       up with one more frame, since the final step is always
                       shown.
    :param interval: Delay between frames in milliseconds.
    :param bare: Draw the region, the coverage and the cutters alone, without
                 axes, grid or titles -- the step counter included.
    """
    validator = SolutionValidator(instance)
    total = solution.max_tour_length
    stride = step if step is not None else math.ceil((total + 1) / max_frames)
    stride = max(1, stride)
    shown = list(range(0, total + 1, stride))
    if shown[-1] != total:
        shown.append(total)

    walks = [_walk(tour, total) for tour in solution.tours]
    # A cutter that is back at its start is standing still, not passing over
    # its spot again and again, so each tour counts its own steps only and
    # never the padding `_walk` adds. The last frame then leaves exactly the
    # counts `create_solution_plot` draws.
    steps = [max(tour.length, 1) for tour in solution.tours]
    swept = validator.swept_cells(solution)
    # The positions the verifier keeps: from anywhere else the cutter cannot
    # touch the region, and the plot's counts leave those placements out too.
    reachable = validator.reachable_anchors(solution)
    palette = colormaps["tab10"]
    # Fixed up front from the finished counts, so that a cell does not change
    # shade as the scale it is measured against grows under it.
    busiest = _scale_max(_visit_counts(validator, solution))
    # The layers share the union's origin and extent, so they line up both with
    # each other and with the blits that fill them in.
    covered = [np.zeros(swept.mask.shape, dtype=np.int64) for _ in solution.tours]

    fig, ax = plt.subplots(figsize=(9, 9))
    images = [
        _draw_counts(
            ax,
            CellCounts(layer, swept.origin),
            palette(index % palette.N),
            busiest,
            alpha=_SWEPT_ALPHA,
            zorder=1,
        )
        for index, layer in enumerate(covered)
    ]
    _draw_region_outline(ax, instance)
    patches = _draw_tours(ax, instance, solution, palette, moving=True)
    outline = _cutter_outline(instance)

    _set_limits(
        ax,
        _union_bounds(
            _region_bounds(instance.region_to_cover),
            _drawn_bounds(swept),
            _tour_bounds(solution),
        ),
    )
    _style_axes(ax, bare=bare)
    _finish(fig, f"CG:SHOP 2027 solution for {instance.instance_uid}", bare=bare, ax=ax)

    stamped_until = -1

    def stamp(first: int, last: int) -> None:
        """Adds every placement in `first .. last` to each cutter's counts."""
        for positions, layer, limit in zip(walks, covered, steps, strict=True):
            walked = positions[first : min(last + 1, limit)]
            if len(walked):
                walked = walked[reachable.contains_points(walked)]
            visited = _visited_anchors(walked)
            if visited is None:
                continue
            painted = dilate_counts(visited, validator.cutter)
            _blit_counts(layer, swept.origin, painted.counts, painted.origin)

    def update(frame: int):
        nonlocal stamped_until
        travelled = shown[frame]
        if travelled < stamped_until:  # the animation looped back to the start
            for layer in covered:
                layer[:] = 0
            stamped_until = -1
        # Every intermediate placement is stamped, so a large stride never
        # makes the coverage lag behind the cutters.
        stamp(stamped_until + 1, travelled)
        stamped_until = travelled
        for image, layer in zip(images, covered, strict=True):
            image.set_data(_as_count_image(layer))
        for patch, positions in zip(patches, walks, strict=True):
            patch.set_xy(_placed(outline, positions[travelled]))
        if not bare:
            ax.set_title(f"step {travelled} / {total}", fontsize=9, pad=4)
        return [*images, *patches]

    return FuncAnimation(fig, update, frames=len(shown), interval=interval, blit=False)


def _visited_anchors(positions: np.ndarray) -> CellCounts | None:
    """
    The given anchor positions as a count grid over their own bounding box,
    a position stood on twice counting twice.
    """
    if not len(positions):
        return None
    x_min, y_min = (int(v) for v in positions.min(axis=0))
    x_max, y_max = (int(v) for v in positions.max(axis=0))
    counts = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.int64)
    np.add.at(counts, (positions[:, 1] - y_min, positions[:, 0] - x_min), 1)
    return CellCounts(counts, (x_min, y_min))
