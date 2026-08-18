"""
Reference geometry derived from an instance alone.

The verifier and the plots both shift the cutter by `cutter_center` before
using it, so an expectation taken from either of them is blind to a wrong sign:
the offset cancels on both sides. Everything here goes back to
`instance.cutter` and `instance.cutter_center` instead, which is what makes it
usable as the expected value in a test about where a cutter ends up.
"""

from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from cgshop2027_pyutils.schemas import CGSHOP2027Instance

__all__ = ["footprint", "footprint_polygon"]


def footprint(instance: CGSHOP2027Instance) -> list[tuple[int, int]]:
    """
    The cutter's cells relative to its anchor, i.e. to its center.

    Cells are picked by their midpoint, which decides membership exactly for a
    rectilinear polyomino and does not go through the rasterizer under test.
    """
    polygon = Polygon(zip(instance.cutter.x, instance.cutter.y, strict=True))
    center_x, center_y = instance.cutter_center
    x_min, y_min, x_max, y_max = (int(value) for value in polygon.bounds)
    return [
        (x - center_x, y - center_y)
        for x in range(x_min, x_max)
        for y in range(y_min, y_max)
        if polygon.contains(box(x, y, x + 1, y + 1).centroid)
    ]


def footprint_polygon(
    instance: CGSHOP2027Instance, position: tuple[int, int] = (0, 0)
) -> Polygon:
    """
    The area the cutter occupies with its center placed at `position`.

    Unlike `SolutionValidator.swept_cells`, this is not clipped to the region's
    surroundings, so it stays meaningful for a placement that misses the region
    entirely.
    """
    return unary_union(
        [
            box(
                x + position[0],
                y + position[1],
                x + position[0] + 1,
                y + position[1] + 1,
            )
            for x, y in footprint(instance)
        ]
    )
