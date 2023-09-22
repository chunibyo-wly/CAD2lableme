import json
import random
import numpy as np
import cv2
import copy
from shapely import (
    MultiPoint,
    Polygon,
    convex_hull,
    within,
    minimum_rotated_rectangle,
    concave_hull,
    LineString,
)
from tqdm import trange, tqdm
from ifcutil import polygon2ifcwall
import subprocess
import ifcopenshell

# opencv 坐标系左上角
# CAD 坐标系左下角
# 横X 竖Y

ox, oy, w, h = 0, 0, 0, 0
scale = 0


class UnionFindSet(object):
    """并查集"""

    def __init__(self, data_list):
        """初始化两个字典，一个保存节点的父节点，另外一个保存父节点的大小
        初始化的时候，将节点的父节点设为自身，size设为1"""
        self.father_dict = {}
        self.size_dict = {}

        for node in data_list:
            self.father_dict[node] = node
            self.size_dict[node] = 1

    def find(self, node):
        """使用递归的方式来查找父节点

        在查找父节点的时候，顺便把当前节点移动到父节点上面
        这个操作算是一个优化
        """
        father = self.father_dict[node]
        if node != father:
            if father != self.father_dict[father]:  # 在降低树高优化时，确保父节点大小字典正确
                self.size_dict[father] -= 1
            father = self.find(father)
        self.father_dict[node] = father
        return father

    def is_same_set(self, node_a, node_b):
        """查看两个节点是不是在一个集合里面"""
        return self.find(node_a) == self.find(node_b)

    def union(self, node_a, node_b):
        """将两个集合合并在一起"""
        if node_a is None or node_b is None:
            return

        a_head = self.find(node_a)
        b_head = self.find(node_b)

        if a_head != b_head:
            a_set_size = self.size_dict[a_head]
            b_set_size = self.size_dict[b_head]
            if a_set_size >= b_set_size:
                self.father_dict[b_head] = a_head
                self.size_dict[a_head] = a_set_size + b_set_size
            else:
                self.father_dict[a_head] = b_head
                self.size_dict[b_head] = a_set_size + b_set_size


def get_axis(items):
    bound = []
    for item in items:
        feature_type = item["json_featuretype"]
        geometry = item["json_geometry"]["type"]
        coordinates = item["json_geometry"]["coordinates"]
        if feature_type == "图框" and geometry == "Polygon":
            coordinates = np.array(coordinates)
            coordinates = coordinates.reshape(-1, 2)
            X, Y = coordinates[:, 0], coordinates[:, 1]
            ox, oy = np.min(X), np.min(Y)
            w, h = np.max(X) - np.min(X), np.max(Y) - np.min(Y)
            return ox, oy, w, h
        elif feature_type == "0" and geometry == "LineString":
            for p in coordinates:
                if len(p) != 3:
                    continue
                bound.append(p)

    if len(bound) != 0:
        bound = np.array(bound)
        X, Y = bound[:, 0], bound[:, 1]
        ox, oy = np.min(X), np.min(Y)
        w, h = np.max(X) - np.min(X), np.max(Y) - np.min(Y)
        return ox, oy, w, h
    assert False, "no 图框 found"


def transform_axis(p):
    if len(p) > 2:
        p = p[:2]
    p = np.array(p)
    p[0] -= ox
    p[1] -= oy
    p[1] = h - p[1]
    x, y = p * scale
    p = np.array([np.around(x), np.around(y)], dtype=np.int32)
    return p


def collect_walls(items):
    walls = []
    for item in items:
        if "json_featuretype" in item:
            feature_type = item["json_featuretype"]
            geometry = item["json_geometry"]["type"]
            coordinates = item["json_geometry"]["coordinates"]
        elif "type" in item and "id" in item:
            feature_type = item["properties"]["Layer"]
            geometry = item["geometry"]["type"]
            coordinates = item["geometry"]["coordinates"]
        else:
            continue
        if (
            "WALL" in feature_type or "COLS" in feature_type
        ) and geometry == "LineString":
            for coordinate in coordinates:
                if len(coordinate) == 2:
                    coordinate.append(0.0)
            coordinates = np.array(coordinates)
            n = len(coordinates)
            for i in range(n - 1):
                p0, p1 = copy.deepcopy(coordinates[i]), copy.deepcopy(
                    coordinates[i + 1]
                )
                if scale != 0:
                    p0, p1 = transform_axis(p0), transform_axis(p1)
                    if isclose(p0[0], p1[0], 1) and isclose(p0[1], p1[1], 1):
                        continue
                else:
                    p0 = np.array([np.around(p0[0], 3), np.around(p0[1], 3)])
                    p1 = np.array([np.around(p1[0], 3), np.around(p1[1], 3)])
                walls.append([p0, p1])
    return np.array(walls)


def collect_doors_and_windows(items):
    doors, segments = [], []
    for item in items:
        if "json_featuretype" in item:
            feature_type = item["json_featuretype"]
            geometry = item["json_geometry"]["type"]
            coordinates = item["json_geometry"]["coordinates"]
        elif "type" in item and "id" in item:
            feature_type = item["properties"]["Layer"]
            geometry = item["geometry"]["type"]
            coordinates = item["geometry"]["coordinates"]
        else:
            continue

        if feature_type in ("WINDOW", "DOOR_FIRE") and geometry in (
            "LineString",
            "Polygon",
        ):
            coordinates = np.array(coordinates).reshape(-1, 2)
            n = len(coordinates)
            tmp = []
            for i in range(n - 1):
                p0, p1 = copy.deepcopy(coordinates[i]), copy.deepcopy(
                    coordinates[i + 1]
                )
                if scale != 0:
                    p0, p1 = transform_axis(p0), transform_axis(p1)
                    if isclose(p0[0], p1[0], 1) and isclose(p0[1], p1[1], 1):
                        continue
                else:
                    p0 = np.array(p0)
                    p1 = np.array(p1)
                if n >= 18:
                    tmp.append([p0, p1])
                elif n >= 2:
                    segments.append([p0, p1])
            if n >= 15:
                doors.append(tmp)
    return doors, segments


def debug_walls(image, walls, door=False, _scale=1.0, offsetX=0, offsetY=0):
    image = image.copy()
    thickness = 3
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    # if door:
    #     a, o, b = np.array(walls)
    #     r = int(np.linalg.norm(a - o))
    #     cv2.ellipse(
    #         image,
    #         o.astype(np.int32),
    #         (r, r),
    #         0,
    #         270,
    #         180,
    #         color,
    #         thickness,
    #     )
    #     return image

    if isinstance(walls, Polygon):
        x, y = np.array(walls.exterior.xy)
        cv2.polylines(
            image,
            [(np.array([x + offsetX, y + offsetY]) / _scale).T.astype(np.int32)],
            True,
            color,
            thickness,
        )
    else:
        tmp = np.array(walls)
        tmp[:, :, 0] += offsetX
        tmp[:, :, 1] += offsetY
        for wall in tmp:
            p0, p1 = np.array(wall)
            # p01 = (p1 - p0) // 10 * 1
            # p0 += p01
            # p1 -= p01
            cv2.line(
                image,
                (int(p0[0] / _scale), int(p0[1] / _scale)),
                (int(p1[0] / _scale), int(p1[1] / _scale)),
                color,
                thickness,
            )
    return image


def group_walls(walls):
    tuple_walls = []
    tuple_point_set = set()
    for wall in walls:
        p0, p1 = wall
        p0, p1 = (p0[0], p0[1]), (p1[0], p1[1])
        tuple_walls.append((p0, p1))
        tuple_point_set.add(p0)
        tuple_point_set.add(p1)

    ufs = UnionFindSet(list(tuple_point_set))

    for wall in tuple_walls:
        p0, p1 = wall
        ufs.union(p0, p1)

    cnt = 0
    father2idx = dict()
    result = []
    for wall in tuple_walls:
        father = ufs.find(wall[0])
        if father not in father2idx:
            father2idx[father] = cnt
            result.append([])
            cnt += 1
        result[father2idx[father]].append(wall)
    return [i for i in result if len(i) >= 4], [i[0] for i in result if len(i) == 1]


def isclose(a, b, eps=1e-4):
    return np.fabs(a - b) < eps


def angle(a, b):
    return (
        np.arccos(np.abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
        * 180
        / np.pi
    )


def colinear(o, a, b):
    x1, y1 = a[0] - o[0], a[1] - o[1]
    x2, y2 = b[0] - o[0], b[1] - o[1]
    return isclose(x1 * y2 - x2 * y1, 0)
    # return angle(np.array([x1, y1]), np.array([x2, y2])) < 3


def perpendicular(o, a, b):
    x1, y1 = a[0] - o[0], a[1] - o[1]
    x2, y2 = b[0] - o[0], b[1] - o[1]
    return isclose(x1 * x2 + y1 * y2, 0, eps=1e-2)


def parallel(wall1, wall2, eps=1e-4):
    p1, p2 = np.array(wall1)
    p3, p4 = np.array(wall2)
    a, b = p2 - p1, p4 - p3
    return angle(a, b) < 5


def walls2concave(walls):
    assert len(walls) > 0

    result = []
    p2p = dict()
    for wall in walls:
        p0, p1 = wall
        if p0 not in p2p:
            p2p[p0] = set()
        p2p[p0].add(p1)
        if p1 not in p2p:
            p2p[p1] = set()
        p2p[p1].add(p0)

    s1 = walls[0][0]
    for p in p2p:
        if len(p2p[p]) == 2:
            a, b = p2p[p]
            if not colinear(p, a, b):
                s1 = p
                break
    for p in p2p:
        if len(p2p[p]) > 2:
            return [], False
        elif len(p2p[p]) == 1:
            s1 = p

    result.append(s1)

    p = s1
    while len(result) < len(p2p):
        for c in p2p[p]:
            if c not in result:
                result.append(c)
                p = c
                break
    result.append(s1)
    return result, len(p2p[s1]) == 2


def groups2concave(groups, auto_link=True):
    concaves = []
    for walls in groups:
        pts, double = walls2concave(walls)
        if auto_link and len(pts) < 4:
            continue
        try:
            if not auto_link and not double:
                if len(pts) >= 4 and (
                    colinear(pts[0], pts[-2], pts[1])
                    or colinear(pts[-2], pts[0], pts[-3])
                    or perpendicular(pts[0], pts[-2], pts[1])
                    or perpendicular(pts[-2], pts[0], pts[-3])
                ):
                    concaves.append(Polygon(pts))
                else:
                    concaves.append(walls)
            else:
                concaves.append(Polygon(pts))
        except Exception as e:
            print(e)
            continue
    return concaves


def group2rect(groups):
    concaves = groups2concave(groups)

    groups = []
    for concave in concaves:
        pts = copy.deepcopy(np.array(concave.exterior.xy).T.tolist()[:-1])
        n = len(pts)
        pts += pts
        groups.append(
            [(tuple(p1), tuple(p2)) for (p1, p2) in zip(pts[:n], pts[1 : n + 1])]
        )

    _groups = []
    for group in groups:
        _i = 0
        tmp = []
        while _i < len(group):
            cur = group[_i]
            next = group[(_i + 1) % len(group)]
            while colinear(next[0], cur[0], next[1]):
                cur = (cur[0], next[1])
                _i += 1
                next = group[(_i + 1) % len(group)]
            _i += 1
            tmp.append(cur)
        _groups.append(tmp)
    groups = copy.deepcopy(_groups)

    for i in range(len(concaves)):
        for j in range(i + 1, len(concaves)):
            if within(concaves[j], concaves[i]):
                concaves[i] = Polygon(
                    np.array(concaves[i].exterior.coords.xy).T,
                    [np.array(concaves[j].exterior.coords.xy).T],
                )
                groups[i] += groups[j]
                groups[j] = []
                break

    result = []
    for idx, walls in enumerate(groups):
        ufs = UnionFindSet(walls)
        base = concaves[idx]

        for i in range(len(walls)):
            for j in range(i + 1, len(walls)):
                # plt.plot([walls[i][0][0], walls[i][1][0]], [walls[i][0][1], walls[i][1][1]])
                # plt.plot([walls[j][0][0], walls[j][1][0]], [walls[j][0][1], walls[j][1][1]])
                # plt.show()
                if (not parallel(walls[i], walls[j], 1e-2)) or colinear(
                    walls[j][0], walls[i][0], walls[j][1]
                ):
                    continue
                polygon = convex_hull(
                    MultiPoint([walls[i][0], walls[i][1], walls[j][0], walls[j][1]])
                )
                if not isinstance(polygon, Polygon) or not polygon.is_valid:
                    continue
                try:
                    if not within(polygon, base):
                        continue
                except Exception as e:
                    print(e)
                    continue
                ufs.union(walls[i], walls[j])

        cnt = 0
        father2idx = dict()
        _result = []
        for wall in walls:
            father = ufs.find(wall)
            if father not in father2idx:
                father2idx[father] = cnt
                _result.append([])
                cnt += 1
            _result[father2idx[father]].append(wall)
        for tmp in _result:
            if len(tmp) > 1:
                result.append(tmp)
    return result, concaves


def group2convex(groups):
    result = []
    for group in groups:
        convex = convex_hull(
            MultiPoint([line[0] for line in group] + [line[1] for line in group])
        )
        result.append(convex)
    return result


def assign_segments2doors(doors, groups):
    doors_dict = dict()
    for door in doors:
        doors_dict[(door[0][0][0], door[0][0][1])] = door
        doors_dict[(door[-1][-1][0], door[-1][-1][1])] = door

    results_door, results_window = [], []
    for group in groups:
        pts = np.array(group.exterior.xy).T.tolist()
        flag = False
        # 四边形上的点
        for rect_idx, pt1 in enumerate(pts):
            pt1 = np.asarray(pt1)
            # 曲线上的点
            for pt2 in doors_dict:
                if flag:
                    break
                door = doors_dict[pt2]
                pt2 = np.asarray(pt2)
                if (
                    np.linalg.norm(pt2 - pt1) <= 200
                    and np.linalg.norm(pt2 - pt1)
                    < np.linalg.norm(pt2 - pts[(rect_idx + 1) % 4])
                    and np.linalg.norm(pt2 - pt1)
                    < np.linalg.norm(pt2 - pts[rect_idx - 1])
                ):
                    flag = True
                    o = None
                    if np.linalg.norm(pts[(rect_idx + 1) % 4] - pt1) > np.linalg.norm(
                        pts[rect_idx - 1] - pt1
                    ):
                        o = pts[(rect_idx + 1) % 4]
                    else:
                        o = pts[rect_idx - 1]
                    door_pts = [i[0] for i in door]
                    door_pts.append(door[-1][-1])
                    r = np.mean([np.linalg.norm(i - o) for i in door_pts])
                    _door_pts = [
                        (i - o) / np.linalg.norm(i - o) * r + o for i in door_pts
                    ]
                    if np.any(door_pts[0] == pt2):
                        _door_pts.reverse()
                    _door_pts.insert(0, np.array(o))
                    # ! 点数目控制，用来验证圆圈的顺序问题, 圆心->墙上的点->远点
                    _door_pts = _door_pts[:]

                    n = len(_door_pts)
                    _door_pts += _door_pts

                    new_door = [
                        (tuple(p1), tuple(p2))
                        for (p1, p2) in zip(_door_pts[:n], _door_pts[1 : n + 1])
                    ]
                    results_door.append(new_door)

        if not flag:
            n = len(pts)
            pts += pts
            results_window.append(
                [(tuple(p1), tuple(p2)) for (p1, p2) in zip(pts[:n], pts[1 : n + 1])]
            )

    return results_door, results_window


def write_component(convexes, label, rectangle=False):
    shapes = []

    template = {
        "label": label,
        "shape_type": "polygon",
        "flags": {},
    }

    if label == "door":
        for convex in convexes:
            door = copy.deepcopy(template)
            door["points"] = np.array([i[0] for i in convex] + [convex[0][0]]).tolist()
            shapes.append(door)
        return shapes

    hash = set()
    for convex in convexes:
        wall = copy.deepcopy(template)
        convex = convex_hull(
            MultiPoint([line[0] for line in convex] + [line[1] for line in convex])
        )
        if convex in hash:
            continue
        hash.add(convex)
        if not isinstance(convex, Polygon):
            continue
        if rectangle:
            convex = minimum_rotated_rectangle(convex)
        wall["points"] = np.array(convex.exterior.xy).T.tolist()
        shapes.append(wall)
    return shapes


def merge_wallwindow_segments(walls, single_segments):
    result = set(single_segments)
    hash = dict()
    for wall in walls:
        for segment in wall:
            if segment[0] not in hash:
                hash[segment[0]] = []
            if segment[1] not in hash:
                hash[segment[1]] = []
            hash[segment[0]].append(segment)
            hash[segment[1]].append(segment)
    for segment in single_segments:
        for p in segment:
            if p in hash:
                wall_segments = hash[p]
                for wall in wall_segments:
                    a, b = np.array(
                        (wall[1][0] - wall[0][0], wall[1][1] - wall[0][1])
                    ), np.array(
                        (segment[1][0] - segment[0][0], segment[1][1] - segment[0][1])
                    )
                    if abs(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))) < 1e-3:
                        result.add(wall)
    windows, _ = group_walls(list(result))
    return [
        [segment for segment in window if segment in single_segments]
        for window in windows
    ]


def get_boundary(shapes):
    pts = [shape["points"] for shape in shapes]
    convex = convex_hull(MultiPoint([j for i in pts for j in i]))
    return np.array(convex.exterior.xy).T.tolist()


def polygon2obj(groups):
    points = []
    faces = []

    height = 5000
    cnt = 1
    for group in groups:
        if isinstance(group, Polygon):
            exterior = np.array(group.exterior.xy).T.tolist()
            n = len(exterior)
            exterior += exterior
            segments = [[exterior[i], exterior[i + 1]] for i in range(n)]
        else:
            segments = group

        for segment in segments:
            x1, y1, x2, y2 = segment[0][0], segment[0][1], segment[1][0], segment[1][1]
            points.append([x1, y1, 0])
            points.append([x1, y1, height])
            points.append([x2, y2, 0])
            points.append([x2, y2, height])
            faces.append([cnt, cnt + 1, cnt + 3, cnt + 2])
            cnt += 4

        if isinstance(group, Polygon):
            n -= 1
            for index, point in enumerate(exterior):
                if index >= n:
                    break
                points.append([point[0], point[1], 0])
                points.append([point[0], point[1], height])
            faces.append([cnt + i * 2 for i in range(n)])
            faces.append([cnt + i * 2 + 1 for i in range(n)])
            cnt += n * 2

    return points, faces


def main2():
    prefix = "B1"
    with open(f"sjg/{prefix}.json", "r", encoding="utf-8") as f:
        items = json.load(f)
    walls = collect_walls(items)
    groups, _ = group_walls(walls)
    concaves = groups2concave(groups, auto_link=False)
    # 取出groups所有的线段，绘制在一张图上
    points, faces = polygon2obj(concaves)
    with open(f"sjg/{prefix}.obj", "w") as f:
        f.write(
            "mtllib out.mtl\nusemtl image\nvt 1.0 0.0 0.0\nvt 0.0 1.0 0.0\nvt 0.0 0.0 0.0\nvt 1.0 1.0 0.0\n"
        )
        for point in points:
            f.write(f"v {point[0] / 1000} {point[1] / 1000} {point[2] / 1000}\n")
        for face in faces:
            if len(face) == 4:
                f.write(f"f {face[0]}/3 {face[1]}/2 {face[2]}/4 {face[3]}/1\n")
            else:
                f.write(f"f {' '.join(map(str, face))}\n")


def main3():
    scale = 10

    input_file = "./input/F1.dwg"
    input_json = input_file.replace(".dwg", "_geo.json")
    subprocess.run(
        [
            "/usr/local/bin/dwgread",
            input_file,
            "-O",
            "GeoJson",
            "-o",
            input_json,
        ],
        stdout=subprocess.PIPE,
        universal_newlines=True,
    )

    with open(input_json, "r", encoding="utf-8") as f:
        items = json.load(f)
        items = items["features"] if "features" in items else items

    walls = collect_walls(items)
    _walls = np.array(walls)
    _maxx, _maxy = np.amax(_walls.reshape(-1, 2), axis=0)
    _minx, _miny = np.amin(_walls.reshape(-1, 2), axis=0)

    width, height = int(_maxx - _minx) // scale, int(_maxy - _miny) // scale

    image = np.zeros((height, width, 3))
    image.fill(255)

    groups, _ = group_walls(walls)
    concaves = groups2concave(groups, auto_link=False)
    _image = image.copy()
    for concave in tqdm(concaves, desc="凹包可视化"):
        _image = debug_walls(
            _image, concave, _scale=scale, offsetX=-_minx, offsetY=-_miny
        )
    cv2.imwrite(f"./debug/origin3.png", _image)
    image[_image != (255, 255, 255)] = 0

    doors, segments = collect_doors_and_windows(items)
    _image = image.copy()
    for door in tqdm(doors, desc="door 可视化"):
        _image = debug_walls(_image, door, _scale=scale, offsetX=-_minx, offsetY=-_miny)
    cv2.imwrite(f"./debug/origin4.png", _image)

    groups, single_segments = group_walls(segments)
    convexes = group2convex(groups)
    doors, windows = assign_segments2doors(doors, convexes)
    windows += merge_wallwindow_segments(walls, single_segments)

    _image = image.copy()
    for window in tqdm(windows, desc="window 可视化"):
        _image = debug_walls(
            _image, window, _scale=scale, offsetX=-_minx, offsetY=-_miny
        )
    for door in tqdm(doors, desc="door 可视化"):
        _image = debug_walls(_image, door, _scale=scale, offsetX=-_minx, offsetY=-_miny)
    cv2.imwrite(f"./debug/origin5.png", _image)

    # ifc = ifcopenshell.template.create()
    # ifc = polygon2ifcwall(ifc, concaves)
    # ifc.write("a.ifc")


def main():
    global ox, oy, w, h, scale

    for idx in range(1, 7):
        floor = f"F{idx}"
        image = cv2.imread(f"./cropped/{floor}.png", cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        with open(f"./input/{floor}.json", "r", encoding="utf-8") as f:
            items = json.load(f)

        # 画幅
        ox, oy, w, h = get_axis(items)
        scale = max(image.shape) / max(w, h)
        print(image.shape, w, h)
        print(image.shape[0] / image.shape[1], h / w, scale)
        # 画幅

        # 墙
        walls = collect_walls(items)
        _image = image.copy()
        _image = debug_walls(_image, walls)
        cv2.imwrite(f"./debug/{floor}_origin.png", _image)

        groups, _ = group_walls(walls)
        rects, concaves = group2rect(groups)
        _image = image.copy()
        for concave in tqdm(concaves, desc="凹包可视化"):
            _image = debug_walls(_image, concave)
        cv2.imwrite(f"./debug/{floor}_concave.png", _image)

        _image = image.copy()
        for rect in tqdm(rects, desc="对应边可视化"):
            _image = debug_walls(_image, rect)
        cv2.imwrite(f"./debug/{floor}_group.png", _image)
        # 墙

        # 门, 部分窗
        doors, segments = collect_doors_and_windows(items)
        groups, single_segments = group_walls(segments)
        _image = image.copy()
        for group in tqdm(groups, desc="opening segments 可视化"):
            _image = debug_walls(_image, group)
        cv2.imwrite(f"./debug/{floor}_opening_group.png", _image)

        convexs = group2convex(groups)
        doors, windows = assign_segments2doors(doors, convexs)
        _image = image.copy()
        for window in tqdm(windows, desc="window 的凸包可视化"):
            _image = debug_walls(_image, window)
        for door in tqdm(doors, desc="door 可视化"):
            _image = debug_walls(_image, door)
        cv2.imwrite(f"./debug/{floor}_opening_result.png", _image)
        # 门, 部分窗

        # 特殊窗户
        additional_windows = merge_wallwindow_segments(rects, single_segments)
        _image = image.copy()
        for window in tqdm(additional_windows, desc="特殊 window 可视化"):
            _image = debug_walls(_image, window)
        cv2.imwrite(f"./debug/{floor}_additional_window.png", _image)
        # 特殊窗户

        # 写文件
        output = {
            "version": "5.0.1",
            "flags": {},
            "imagePath": f"{floor}.png",
            "imageHeight": h,
            "imageWidth": w,
            "fillColor": [255, 0, 0, 128],
            "lineColor": [0, 255, 0, 128],
            "shapes": [],
            "imageData": None,
        }

        output["shapes"] += write_component(rects, "wall", True)
        output["shapes"] += write_component(
            windows + additional_windows, "window", True
        )
        output["shapes"] += write_component(doors, "door")
        # output["shapes"].append(
        #     {
        #         "label": "boundary",
        #         "shape_type": "polygon",
        #         "flags": {},
        #         "points": get_boundary(output["shapes"]),
        #     }
        # )

        output["boundary"] = get_boundary(output["shapes"])

        with open(f"./output/{floor}.json", "w", encoding="utf-8") as f:
            json.dump(output, f)
        cv2.imwrite(f"./output/{floor}.png", image)


if __name__ == "__main__":
    main3()
