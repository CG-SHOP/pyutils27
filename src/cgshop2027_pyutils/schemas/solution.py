import sys
from collections.abc import Iterator
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from .instance import Coordinate, CoordinateList

Point = tuple[Coordinate, Coordinate]


class CutterTour(BaseModel):
    """
    This schema represents the closed tour of a single cutter, given as the
    sequence of corner points its center passes through.

    The tour is *implicitly* closed: the cutter travels from the last point
    back to the first one. Consecutive points -- the closing pair included --
    must be axis-parallel, i.e., differ in exactly one coordinate; a point may
    well be visited again later, but not immediately again.

    Every point is a position of the cutter's *center*: the footprint at a
    position `p` is the instance's cutter shape translated so that its
    `cutter_center` lands on `p`. A tour of a single point is a cutter that
    stays where it is.
    """

    x: CoordinateList = Field(..., description="X coordinates of the tour's points.")
    y: CoordinateList = Field(..., description="Y coordinates of the tour's points.")

    @model_validator(mode="after")
    def _check_matching_lengths(self) -> Self:
        if len(self.x) != len(self.y):
            msg = (
                f"The coordinate lists must have the same length, but got "
                f"{len(self.x)} x-coordinates and {len(self.y)} y-coordinates."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_for_minimum_points(self) -> Self:
        if not self.x:
            msg = "A tour must consist of at least one point."
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_edges(self) -> Self:
        """
        Rejects edges that are not axis-parallel, and points that repeat the
        point right before them -- which the closing edge extends to a last
        point that repeats the first one.
        """
        last = len(self.x) - 1
        for index, (start, end) in enumerate(self.edges()):
            closing = " closing" if index == last else ""
            if start == end:
                successor = (
                    "first point" if index == last else f"point at index {index + 1}"
                )
                msg = (
                    f"The point {start} at index {index} is immediately repeated "
                    f"by the {successor}; the tour is closed implicitly, "
                    f"and a point may only be revisited later on."
                )
                raise ValueError(msg)
            if start[0] != end[0] and start[1] != end[1]:
                msg = (
                    f"The{closing} edge from {start} to {end} is not orthogonal. "
                    f"All edges must be either horizontal or vertical."
                )
                raise ValueError(msg)
        return self

    @property
    def start(self) -> Point:
        """
        The first point of the tour.
        """
        return (self.x[0], self.y[0])

    @property
    def length(self) -> int:
        """
        The distance the center travels once around the tour, the way back from
        the last point to the first one included.
        """
        return sum(
            abs(end[0] - start[0]) + abs(end[1] - start[1])
            for start, end in self.edges()
        )

    def points(self) -> Iterator[Point]:
        """
        Yields the corner points of the tour, each one exactly once.
        """
        yield from zip(self.x, self.y, strict=True)

    def edges(self) -> Iterator[tuple[Point, Point]]:
        """
        Yields the edges of the tour as pairs of points, the closing edge from
        the last point back to the first one included; hence one edge per
        point, and none at all for a tour of a single point.
        """
        if len(self.x) < 2:
            return
        points = list(self.points())
        yield from zip(points, [*points[1:], points[0]], strict=True)

    def positions(self) -> Iterator[Point]:
        """
        Yields the position of the cutter's center after every unit of travel,
        starting and ending at `start`; hence `length + 1` positions.
        """
        x, y = self.start
        yield (x, y)
        for start, end in self.edges():
            dx = (end[0] > start[0]) - (end[0] < start[0])
            dy = (end[1] > start[1]) - (end[1] < start[1])
            for _ in range(abs(end[0] - start[0]) + abs(end[1] - start[1])):
                x += dx
                y += dy
                yield (x, y)


class CGSHOP2027Solution(BaseModel):
    """
    This schema represents a solution for the CG:SHOP 2027 challenge.
    The objective is `max_tour_length`, the length of the longest tour, and is
    to be minimized.
    """

    content_type: Literal["CGSHOP2027_Solution"] = "CGSHOP2027_Solution"

    instance_uid: str = Field(..., description="Unique identifier of the instance.")

    tours: list[CutterTour] = Field(
        ..., description="The closed tour of each cutter used by this solution."
    )

    meta: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata about the solution, e.g., the name of the algorithm used to compute it.",
    )

    @property
    def max_tour_length(self) -> int:
        """
        The length of the longest tour, i.e., the objective value.
        """
        return max((tour.length for tour in self.tours), default=0)

    @property
    def objective(self) -> int:
        """
        The value to be minimized, which is `max_tour_length`.
        """
        return self.max_tour_length

    @property
    def total_tour_length(self) -> int:
        """
        The distance travelled by all cutters together.
        """
        return sum(tour.length for tour in self.tours)
