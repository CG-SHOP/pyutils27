import sys
from typing import Literal

from pydantic import BaseModel, Field, PositiveInt, model_validator
from shapely.geometry import LinearRing, Polygon
from shapely.geometry import Point as _ShapelyPoint
from shapely.validation import explain_validity

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


Coordinate = int
CoordinateList = list[Coordinate]


class PointSequence(BaseModel):
    """
    This schema represents a sequence of points in 2D space;
    it is *implicitly* closed and must be a simple polyomino;
    it can be either CW or CCW-oriented, and must not self-intersect.
    """

    x: CoordinateList = Field(..., description="X coordinates of the points.")
    y: CoordinateList = Field(..., description="Y coordinates of the points.")

    def is_ccw(self) -> bool:
        """
        Determines if the sequence of points is oriented counter-clockwise (CCW).
        """
        area = 0
        prev = (self.x[-1], self.y[-1])
        for point in zip(self.x, self.y, strict=True):
            area += (point[0] - prev[0]) * (point[1] + prev[1])
            prev = point
        return area < 0

    @model_validator(mode="after")
    def _check_for_duplicate_points(self) -> Self:
        """
        Rejects sequences in which the same point occurs more than once.
        """
        if len(self.x) != len(self.y):
            msg = (
                f"The coordinate lists must have the same length, but got "
                f"{len(self.x)} x-coordinates and {len(self.y)} y-coordinates."
            )
            raise ValueError(msg)
        seen: set[tuple[Coordinate, Coordinate]] = set()
        for index, point in enumerate(zip(self.x, self.y, strict=True)):
            if point in seen:
                msg = f"The point {point} at index {index} is a duplicate."
                raise ValueError(msg)
            seen.add(point)
        return self

    @model_validator(mode="after")
    def _check_for_minimum_points(self) -> Self:
        """
        Rejects sequences that contain fewer than 4 points.
        """
        if len(self.x) < 4:
            msg = (
                f"A valid polyomino must have at least 4 points, but got "
                f"{len(self.x)} points."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_for_orthogonal_edges(self) -> Self:
        prev = (self.x[-1], self.y[-1])
        for point in zip(self.x, self.y, strict=True):
            if prev[0] != point[0] and prev[1] != point[1]:
                msg = (
                    f"The edge from {prev} to {point} is not orthogonal. "
                    f"All edges must be either horizontal or vertical."
                )
                raise ValueError(msg)
            prev = point
        return self

    @model_validator(mode="after")
    def _check_for_self_intersection(self) -> Self:
        linear_ring = LinearRing(zip(self.x, self.y, strict=True))
        if not linear_ring.is_valid:
            msg = f"One component of the polyomino is self-intersecting: {explain_validity(linear_ring)}"
            raise ValueError(msg)
        return self


class PolyominoWithHoles(BaseModel):
    """
    This schema represents a polyomino with holes.

    The polyomino consists of an outer boundary and zero or more
    inner boundaries (holes). Each boundary is represented as a
    pair of lists of integer x and y coordinates, which define the
    vertices of the polygon in CCW order (outer boundary) or CW order
    (inner boundaries).
    """

    outer_boundary: PointSequence = Field(
        ..., description="Outer boundary of the polyomino in CCW order."
    )
    inner_boundaries: list[PointSequence] = Field(
        ..., description="Inner boundaries (holes) of the polyomino in CW order."
    )

    @model_validator(mode="after")
    def _check_orientations(self) -> Self:
        """
        Validates the orientations of the outer and inner boundaries.
        """
        if not self.outer_boundary.is_ccw():
            msg = "The outer boundary must be oriented counter-clockwise (CCW)."
            raise ValueError(msg)
        for index, inner_boundary in enumerate(self.inner_boundaries):
            if inner_boundary.is_ccw():
                msg = f"The inner boundary at index {index} must be oriented clockwise (CW)."
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_polygon_validity(self) -> Self:
        """
        Validates the polygon formed by the outer and inner boundaries.
        """
        outer_coords = list(
            zip(self.outer_boundary.x, self.outer_boundary.y, strict=True)
        )
        inner_coords_list = [
            list(zip(inner_boundary.x, inner_boundary.y, strict=True))
            for inner_boundary in self.inner_boundaries
        ]
        polygon = Polygon(outer_coords, holes=inner_coords_list)
        if not polygon.is_valid:
            msg = f"The polygon formed by the outer and inner boundaries is invalid: {explain_validity(polygon)}"
            raise ValueError(msg)
        return self


class CGSHOP2027Instance(BaseModel):
    """
    This schema represents an instance for the CG:SHOP 2027 challenge.

    Only `content_type` and `instance_uid` are fixed at this point; the fields
    describing the actual problem still have to be added once the challenge is
    decided.
    """

    content_type: Literal["CGSHOP2027_Instance"] = "CGSHOP2027_Instance"

    instance_uid: str = Field(..., description="Unique identifier of the instance.")

    region_to_cover: PolyominoWithHoles = Field(
        ...,
        description="The region that needs to be covered, represented as a polyomino with holes.",
    )

    cutter: PointSequence = Field(
        ...,
        description="The cutter shape that will be used to cover the region, "
        "represented as a sequence of points.",
    )

    cutter_center: tuple[Coordinate, Coordinate] = Field(
        ...,
        description="The center of the cutter shape, represented as a tuple of (x, y) coordinates.",
    )

    number_of_cutters: PositiveInt = Field(
        ...,
        description="The number of cutter shapes available for covering the region.",
    )

    @model_validator(mode="after")
    def _check_cutter_orientation(self) -> Self:
        """
        Validates the orientation of the cutter shape.
        """
        if not self.cutter.is_ccw():
            msg = "The cutter shape must be oriented counter-clockwise (CCW)."
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_cutter_center_within_cutter(self) -> Self:
        """
        Validates that the cutter center lies on or inside the cutter shape.

        The cutter is treated as closed, so a center on an edge or a corner is
        legal -- a 1x1 cutter has no interior lattice point at all.
        """
        cutter_polygon = Polygon(zip(self.cutter.x, self.cutter.y, strict=True))
        if not cutter_polygon.covers(_ShapelyPoint(self.cutter_center)):
            msg = "The cutter center must be within the cutter shape."
            raise ValueError(msg)
        return self
