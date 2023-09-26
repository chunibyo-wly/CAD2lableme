import FreeCAD, Draft, Arch, exportIFC, Part
import pickle
import numpy as np
from tqdm import tqdm
from shapely import Polygon
import sys

doc = FreeCAD.newDocument()

HEIGHT = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
OBJS = []


def find_longest_middle_line(rectangle: Polygon):
    # Get the coordinates of the rectangle vertices
    coords = np.array(rectangle.exterior.coords)
    a = (coords[0] + coords[1]) / 2
    b = (coords[1] + coords[2]) / 2
    c = (coords[2] + coords[3]) / 2
    d = (coords[3] + coords[0]) / 2

    if np.linalg.norm(a - c) > np.linalg.norm(b - d):
        return a, c
    else:
        return b, d


def angle_with_x_axis(p1, p2):
    p1, p2 = np.array(p1), np.array(p2)
    n1 = np.array([1, 0])
    n2 = p2 - p1
    angle = np.arccos(np.dot(n2, n1) / (np.linalg.norm(n2))) * 180 / np.pi
    return angle


def clockwise(p1, p2, p3):
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    return np.cross(p2 - p1, p3 - p1) < 0


def addWall(polygon: Polygon):
    pts = []
    X, Y = np.array(polygon.exterior.xy)
    for x, y in zip(X, Y):
        pts.append(FreeCAD.Vector(x, y, 0.0))

    # wire -> sketch -> face -> wall
    wire = Draft.makeWire(pts, face=True, closed=True)
    doc.recompute()
    sketch = Draft.make_sketch(wire.Shape, autoconstraints=True)
    doc.recompute()
    face = doc.addObject("Part::Face", "Face")
    face.Sources = sketch
    doc.recompute()
    wall = Arch.makeWall(face, height=HEIGHT)
    doc.recompute()

    OBJS.append(wall)
    return wall


def addWallFromLine(p1, p2, width):
    pa = FreeCAD.Vector(p1[0], p1[1], 0)
    pb = FreeCAD.Vector(p2[0], p2[1], 0)
    points = [pa, pb]
    line = Draft.makeWire(points, closed=False, face=False, support=None)
    wall = Arch.makeWall(line, length=None, width=width, height=HEIGHT)
    doc.recompute()

    OBJS.append(wall)
    return wall


def addWindow(polygon: Polygon):
    wall = addWall(polygon)
    p1, p2 = find_longest_middle_line(polygon)

    width = np.linalg.norm(p1 - p2) * 0.95
    height = HEIGHT * 0.5

    placement = FreeCAD.Placement(
        FreeCAD.Vector(-width / 2, 0, 0),
        FreeCAD.Rotation(FreeCAD.Vector(1, 0, 0), 90),
    )
    window = Arch.makeWindowPreset(
        "Open 1-pane",
        width=width,
        height=height,
        h1=1,
        h2=1,
        h3=1,
        w1=1,
        w2=1,
        o1=1,
        o2=1,
        placement=placement,
    )
    doc.recompute()

    angle = angle_with_x_axis(p1, p2)

    Draft.rotate(window, angle)
    Draft.move(
        window,
        FreeCAD.Vector((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, (HEIGHT - height) / 2),
    )

    window.HoleWire = 1
    if wall is not None:
        window.Hosts = wall

    OBJS.append(window)
    return window


def addDoor(door_segments, door_width):
    """
    p1: 圆心
    p2: 靠墙点
    p3: 远墙点
    """
    p1, p2, p3 = np.array(door_segments)[:3, 0]
    wall = addWallFromLine(p1, p2, door_width)

    width = np.linalg.norm(p2 - p1) * 0.95
    height = HEIGHT * 0.7

    placement = FreeCAD.Placement(
        FreeCAD.Vector(-width / 2, 0, 0),
        FreeCAD.Rotation(FreeCAD.Vector(1, 0, 0), 90),
    )
    door = Arch.makeWindowPreset(
        "Simple door",
        width=width,
        height=height,
        h1=1,
        h2=1,
        h3=1,
        w1=1,
        w2=1,
        o1=1,
        o2=1,
        placement=placement,
    )

    angle = angle_with_x_axis(p1, p2)

    Draft.rotate(door, angle)
    Draft.move(
        door,
        FreeCAD.Vector((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, 0),
    )
    door.HoleWire = 1
    door.SymbolPlan = True
    door.Opening = 50
    if wall is not None:
        door.Hosts = wall

    doc.recompute()

    OBJS.append(door)
    return door


def addSlab(x1, y1, x2, y2):
    height = 100

    pts = [
        FreeCAD.Vector(x1, y1, -height),
        FreeCAD.Vector(x2, y1, -height),
        FreeCAD.Vector(x2, y2, -height),
        FreeCAD.Vector(x1, y2, -height),
    ]

    wire = Draft.makeWire(pts, face=True, closed=True)
    doc.recompute()
    sketch = Draft.make_sketch(wire.Shape, autoconstraints=True)
    doc.recompute()
    slab = Arch.makeStructure(sketch)
    Draft.autogroup(slab)
    doc.recompute()
    slab.Height = height
    slab.IfcType = "Slab"
    slab.Label = "Slab"
    doc.recompute()

    OBJS.append(slab)
    return slab


def main():
    if len(sys.argv) > 2:
        file = sys.argv[2]
    else:
        file = "F1"

    with open(f"{file}", "rb") as f:
        data = pickle.load(f)

    # 添加墙
    walls = data["walls"]
    for wall in tqdm(walls):
        if not isinstance(wall, Polygon):
            continue
        addWall(wall)

    # 添加地板
    # TODO: 建立墙和地板的关联
    slab = data["slab"]
    addSlab(*slab)

    # 添加窗户
    windows = data["windows"]
    for window in tqdm(windows):
        if not isinstance(window, Polygon):
            continue
        addWindow(window)

    # 添加门
    doors = data["doors"]
    doors_width = data["doors_width"]
    for door, door_width in tqdm(zip(doors, doors_width)):
        if door_width < 0:
            continue
        addDoor(door, door_width)

    doc.recompute()

    file = ".".join(file.split(".")[:-1])

    doc.saveAs(f"{file}.FCStd")
    exportIFC.export(OBJS, f"{file}.ifc")


main()
