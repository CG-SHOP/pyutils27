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
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.patches import PathPatch
from matplotlib.patches import Polygon as PolygonPatch
from matplotlib.path import Path as MplPath

from .grid import CellSet, _blit, dilate
from .schemas import CGSHOP2027Instance, CGSHOP2027Solution, CutterTour
from .schemas.instance import PointSequence, PolyominoWithHoles
from .verify import SolutionValidator

__all__ = [
    "create_instance_plot",
    "create_solution_animation",
    "create_solution_plot",
]

_REGION_FACE = "#d9d9d9"
_REGION_EDGE = "#404040"
_SWEPT = "#7fbf7f"
_UNCOVERED = "#d62728"
_CUTTER = "#1f77b4"


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


def _draw_cells(
    ax: Axes, cells: CellSet, color: str, alpha: float = 1.0, zorder: float = 1.0
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


def _as_image(mask: np.ndarray) -> np.ma.MaskedArray:
    """Masks out the empty cells so that only the set ones are painted."""
    return np.ma.masked_where(~mask, np.ones_like(mask, dtype=np.uint8))


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

    :param bare: Draw the region, the coverage and the tours alone, without
                 axes, grid or titles -- the verdict included, so a gap then
                 shows only as the red it is drawn in.
    """
    validator = SolutionValidator(instance)
    errors = validator.check_for_errors(solution)
    swept = validator.swept_cells(solution)
    uncovered = validator.uncovered_cells(solution)

    fig, ax = plt.subplots(figsize=(9, 9))
    # The swept area is not clipped to the region, so effort spent just outside
    # it is visible too. It is clipped to the region's neighbourhood by the
    # verifier though, so a longer excursion shows only as a trajectory.
    _draw_cells(ax, swept, _SWEPT, alpha=0.55, zorder=1)
    _draw_cells(ax, uncovered, _UNCOVERED, alpha=0.85, zorder=2)
    ax.add_patch(
        PathPatch(
            _polyomino_path(instance.region_to_cover),
            facecolor="none",
            edgecolor=_REGION_EDGE,
            linewidth=1.4,
            zorder=3,
        )
    )

    palette = colormaps["tab10"]
    outline = _cutter_outline(instance)
    for index, tour in enumerate(solution.tours):
        color = palette(index % palette.N)
        ring = _closed(tour)
        ax.plot(
            [x for x, _ in ring],
            [y for _, y in ring],
            color=color,
            linewidth=1.2,
            zorder=4,
        )
        ax.plot(
            [tour.start[0]],
            [tour.start[1]],
            marker="o",
            color=color,
            markersize=5,
            zorder=5,
        )
        ax.add_patch(
            PolygonPatch(
                _placed(outline, tour.start),
                closed=True,
                facecolor="none",
                edgecolor=color,
                linestyle="--",
                linewidth=1.0,
                zorder=5,
            )
        )

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
    swept = validator.swept_cells(solution)
    covered = np.zeros_like(swept.mask)

    fig, ax = plt.subplots(figsize=(9, 9))
    image = _draw_cells(
        ax, CellSet(covered, swept.origin), _SWEPT, alpha=0.55, zorder=1
    )
    ax.add_patch(
        PathPatch(
            _polyomino_path(instance.region_to_cover),
            facecolor="none",
            edgecolor=_REGION_EDGE,
            linewidth=1.4,
            zorder=3,
        )
    )

    palette = colormaps["tab10"]
    outline = _cutter_outline(instance)
    patches = []
    for index, tour in enumerate(solution.tours):
        color = palette(index % palette.N)
        ring = _closed(tour)
        ax.plot(
            [x for x, _ in ring],
            [y for _, y in ring],
            color=color,
            linewidth=1.0,
            alpha=0.45,
            zorder=2,
        )
        patch = PolygonPatch(
            _placed(outline, tour.start),
            closed=True,
            facecolor=color,
            edgecolor=_REGION_EDGE,
            alpha=0.75,
            linewidth=1.0,
            zorder=4,
        )
        ax.add_patch(patch)
        patches.append(patch)

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
        """Adds every placement in `first .. last` to the covered area."""
        for positions in walks:
            anchors = _anchor_cells(positions[first : last + 1])
            if anchors is None:
                continue
            painted = dilate(anchors, validator.cutter)
            _blit(covered, swept.origin, painted.mask, painted.origin)

    def update(frame: int):
        nonlocal stamped_until
        travelled = shown[frame]
        if travelled < stamped_until:  # the animation looped back to the start
            covered[:] = False
            stamped_until = -1
        # Every intermediate placement is stamped, so a large stride never
        # makes the coverage lag behind the cutters.
        stamp(stamped_until + 1, travelled)
        stamped_until = travelled
        image.set_data(_as_image(covered))
        for patch, positions in zip(patches, walks, strict=True):
            patch.set_xy(_placed(outline, positions[travelled]))
        if not bare:
            ax.set_title(f"step {travelled} / {total}", fontsize=9, pad=4)
        return [image, *patches]

    return FuncAnimation(fig, update, frames=len(shown), interval=interval, blit=False)


def _anchor_cells(positions: np.ndarray) -> CellSet | None:
    """The given anchor positions as a cell set over their own bounding box."""
    if not len(positions):
        return None
    x_min, y_min = (int(v) for v in positions.min(axis=0))
    x_max, y_max = (int(v) for v in positions.max(axis=0))
    mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=bool)
    mask[positions[:, 1] - y_min, positions[:, 0] - x_min] = True
    return CellSet(mask, (x_min, y_min))
