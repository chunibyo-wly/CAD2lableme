import typing
from dataclasses import dataclass, field

import ifcopenshell
import ifcopenshell.template
from shapely import Polygon
import numpy as np


@dataclass
class rect:
    width: float
    height: float

    def build(self, f):
        return f.createIfcRectangleProfileDef(
            "AREA", None, None, self.width, self.height
        )


@dataclass
class circle:
    radius: float

    def build(self, f):
        return f.createIfcCircleProfileDef("AREA", None, None, self.radius)


@dataclass
class opening:
    x: float
    z: float
    shape: typing.Any
    depth: float
    direc: tuple = field(default_factory=lambda: (0.0, 0.0, -1.0))


_O = 0.0, 0.0, 0.0
_X = 1.0, 0.0, 0.0
_Y = 0.0, 1.0, 0.0
_Z = 0.0, 0.0, 1.0


# Creates an IfcAxis2Placement3D from Location, Axis and RefDirection specified as Python tuples
def create_ifcaxis2placement(f, point=_O, dir1=_Z, dir2=_X):
    point = f.createIfcCartesianPoint(point)
    dir1 = f.createIfcDirection(dir1)
    dir2 = f.createIfcDirection(dir2)
    axis2placement = f.createIfcAxis2Placement3D(point, dir1, dir2)
    return axis2placement


# Creates an IfcLocalPlacement from Location, Axis and RefDirection, specified as Python tuples, and relative placement
def create_ifclocalplacement(f, point=_O, dir1=_Z, dir2=_X, relative_to=None):
    axis2placement = create_ifcaxis2placement(f, point, dir1, dir2)
    ifclocalplacement2 = f.createIfcLocalPlacement(relative_to, axis2placement)
    return ifclocalplacement2


# Creates an IfcPolyLine from a list of points, specified as Python tuples
def create_ifcpolyline(f, point_list):
    ifcpts = []
    for point in point_list:
        point = f.createIfcCartesianPoint(point)
        ifcpts.append(point)
    polyline = f.createIfcPolyLine(ifcpts)
    return polyline


# Creates an IfcExtrudedAreaSolid from a list of points, specified as Python tuples
def create_ifcextrudedareasolid(
    f, point_list, ifcaxis2placement, extrude_dir, extrusion
):
    polyline = create_ifcpolyline(f, point_list)
    ifcclosedprofile = f.createIfcArbitraryClosedProfileDef("AREA", None, polyline)
    ifcdir = f.createIfcDirection(extrude_dir)
    ifcextrudedareasolid = f.createIfcExtrudedAreaSolid(
        ifcclosedprofile, ifcaxis2placement, ifcdir, extrusion
    )
    return ifcextrudedareasolid


def polygon2ifcwall(f, concaves):
    owner_history = f.by_type("IfcOwnerHistory")[0]
    project = f.by_type("IfcProject")[0]
    context = f.by_type("IfcGeometricRepresentationContext")[0]

    wall_placement = create_ifclocalplacement(f, relative_to=None)
    extrusion_placement = create_ifcaxis2placement(f, _O, _Z, _X)

    for concave in concaves:
        if isinstance(concave, Polygon):
            exterior = np.array(concave.exterior.xy).T.tolist()
            point_list_extrusion_area = [(*i, 0.0) for i in exterior]
            solid = create_ifcextrudedareasolid(
                f, point_list_extrusion_area, extrusion_placement, _Z, 4000.0
            )
            body_representation = f.createIfcShapeRepresentation(
                context, "Body", "SweptSolid", [solid]
            )

            product_shape = f.createIfcProductDefinitionShape(
                None, None, [body_representation]
            )

            wall = f.createIfcWallStandardCase(
                ifcopenshell.guid.new(),
                owner_history,
                "Wall",
                None,
                None,
                wall_placement,
                product_shape,
                None,
            )
            f.add(wall)

    return f