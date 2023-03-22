from shapely import Polygon, Point, LineString, within
import matplotlib.pyplot as plt


def show(objs):
    for obj in objs:
        if isinstance(obj, LineString):
            x, y = obj.xy
        elif isinstance(obj, Polygon):
            x, y = obj.exterior.xy
        plt.plot(x, y)
    plt.show()


def test1():
    # True
    polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    line = LineString([[0, 0], [1, 1]])
    print(within(line, polygon))
    show([polygon, line])


def test2():
    # False
    polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    line = LineString([[0, 0], [2, 2]])
    print(within(line, polygon))
    show([polygon, line])


def test3():
    # False, True
    polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    line = LineString([[0, 0], [0, 1]])
    print(within(line, polygon), line.intersects(polygon), line.crosses(polygon))
    show([polygon, line])


def test4():
    # True
    polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    line = LineString([[0.25, 0.25], [0.5, 0.5]])
    print(within(line, polygon))
    show([polygon, line])


def test5():
    # False, True
    polygon = Polygon([(0, 0), (0, 3), (2, 3), (2, 2), (1, 2), (1, 1), (2, 1), (2, 0)])
    line1 = LineString([(2, 0), (2, 3)])
    line2 = LineString([(1, 0), (1, 3)])
    print(within(line1, polygon), line1.intersection(polygon), line1.crosses(polygon))
    print(within(line2, polygon), line2.intersection(polygon), line2.crosses(polygon))
    show([polygon, line1, line2])


def test6():
    polygon1 = Polygon([(0, 0), (0, 3), (2, 3), (2, 2), (1, 2), (1, 1), (2, 1), (2, 0)])
    polygon2 = Polygon([(1, 1), (1, 2), (2, 2), (2, 1)])
    polygon3 = Polygon([(0, 0), (1, 1), (1, 2), (0, 3)])
    polygon4 = Polygon([(0, 0), (2, 1), (2, 2), (0, 3)])

    print(polygon1.intersection(polygon2))
    print(
        within(polygon3, polygon1),
        within(polygon2, polygon1),
        within(polygon4, polygon1),
    )
    show([polygon1, polygon2, polygon3, polygon4])


def main():
    # test1()
    # test2()
    # test3()
    # test4()
    # test5()
    test6()


if __name__ == "__main__":
    main()
