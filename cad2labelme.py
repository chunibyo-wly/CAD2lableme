import json
import random
import numpy as np
import cv2
import copy
from shapely import MultiPoint, Polygon, convex_hull, within
from tqdm import trange, tqdm

# opencv 坐标系左上角
# CAD 坐标系左下角
# 横X 竖Y

ox, oy, w, h = 0, 0, 0, 0
scale = 0
point2idx = {}


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
    for item in items:
        feature_type = item["json_featuretype"]
        geometry = item["json_geometry"]["type"]
        coordinates = np.array(item["json_geometry"]["coordinates"])
        if feature_type == "图框" and geometry == "Polygon":
            coordinates = coordinates.reshape(-1, 2)
            X, Y = coordinates[:, 0], coordinates[:, 1]
            ox, oy = np.min(X), np.min(Y)
            w, h = np.max(X) - np.min(X), np.max(Y) - np.min(Y)
            return ox, oy, w, h
    assert False, "no 图框 found"


def transform_axis(p):
    if len(p) > 2:
        p = p[:2]
    p[0] -= ox
    p[1] -= oy
    p[1] = h - p[1]
    x, y = p * scale
    p = np.array([np.around(x), np.around(y)], dtype=np.int32)
    return p


def collect_walls(items):
    walls = []
    for item in items:
        feature_type = item["json_featuretype"]
        geometry = item["json_geometry"]["type"]
        coordinates = np.array(item["json_geometry"]["coordinates"])
        if (
            feature_type == "WALL"
            and geometry == "LineString"
            and len(coordinates) == 2
        ):
            p0, p1 = coordinates[0], coordinates[1]
            p0, p1 = transform_axis(p0), transform_axis(p1)
            walls.append([p0, p1])
    return np.array(walls)


def debug_walls(image, walls):
    image = image.copy()
    thickness = 3
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    if isinstance(walls, Polygon):
        cv2.polylines(
            image,
            [np.array(walls.exterior.xy).T.astype(np.int32)],
            True,
            color,
            thickness,
        )
    else:
        for wall in walls:
            p0, p1 = wall
            cv2.line(image, (p0[0], p0[1]), (p1[0], p1[1]), color, thickness)
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
    return result


def isclose(a, b, eps=1e-4):
    return np.fabs(a - b) < eps


def colinear(o, a, b):
    x1, y1 = a[0] - o[0], a[1] - o[1]
    x2, y2 = b[0] - o[0], b[1] - o[1]
    return isclose(x1 * y2 - x2 * y1, 0)


def parallel(wall1, wall2):
    p1, p2 = np.array(wall1)
    p3, p4 = np.array(wall2)
    x1, y1 = p2 - p1
    x2, y2 = p4 - p3
    return isclose(x1 * y2 - x2 * y1, 0)


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

    for p in p2p:
        if len(p2p[p]) != 2:
            return []

    s1 = walls[0][0]
    result.append(s1)

    p = s1
    while len(result) < len(p2p):
        for c in p2p[p]:
            if c not in result:
                result.append(c)
                p = c
                break
    result.append(walls[0][0])
    return result


def group2rect(groups):
    concaves = []
    for walls in tqdm(groups):
        pts = walls2concave(walls)
        concaves.append(Polygon(pts))

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
    for i, walls in tqdm(enumerate(groups)):
        ufs = UnionFindSet(walls)
        base = concaves[i]

        for i in range(len(walls)):
            for j in range(i + 1, len(walls)):
                if not parallel(walls[i], walls[j]) or colinear(
                    walls[i][0], walls[j][0], walls[j][1]
                ):
                    continue
                polygon = convex_hull(
                    MultiPoint([walls[i][0], walls[i][1], walls[j][0], walls[j][1]])
                )
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


def main():
    global ox, oy, w, h, scale

    floor = "F1"

    image = cv2.imread(f"./cropped/{floor}.png", cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    with open(f"./input/{floor}.json", "r", encoding="utf-8") as f:
        items = json.load(f)
    ox, oy, w, h = get_axis(items)
    scale = max(image.shape) / max(w, h)
    print(image.shape, w, h)
    print(image.shape[0] / image.shape[1], h / w, scale)

    walls = collect_walls(items)
    _image = image.copy()
    _image = debug_walls(_image, walls)
    cv2.imwrite(f"./debug/{floor}_origin.png", _image)
    groups = group_walls(walls)

    rects, concaves = group2rect(groups)
    _image = image.copy()
    for concave in tqdm(concaves):
        _image = debug_walls(_image, concave)
    cv2.imwrite(f"./debug/{floor}_concave.png", _image)

    _image = image.copy()
    for rect in tqdm(rects):
        _image = debug_walls(_image, rect)
    cv2.imwrite(f"./debug/{floor}_group.png", _image)

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
    wall_template = {
        "label": "wall",
        "shape_type": "polygon",
        "flags": {},
    }
    for rect in tqdm(rects):
        wall = copy.deepcopy(wall_template)
        convex = convex_hull(
            MultiPoint([line[0] for line in rect] + [line[1] for line in rect])
        )
        if not isinstance(convex, Polygon):
            continue
        wall["points"] = np.array(convex.exterior.xy).T.tolist()
        output["shapes"].append(wall)
    with open(f"./output/{floor}.json", "w", encoding="utf-8") as f:
        json.dump(output, f)
    cv2.imwrite(f"./output/{floor}.png", image)


if __name__ == "__main__":
    main()
