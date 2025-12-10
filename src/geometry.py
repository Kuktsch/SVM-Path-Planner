from collections.abc import Iterable
import math

Point = tuple[float, float]


class GeometryUtils:
    """Утилиты для работы с геометрией"""
    
    @staticmethod
    def dot(a: Point, b: Point) -> float:
        return a[0] * b[0] + a[1] * b[1]
    
    @staticmethod
    def sub(a: Point, b: Point) -> Point:
        return (a[0] - b[0], a[1] - b[1])
    
    @staticmethod
    def add(a: Point, b: Point) -> Point:
        return (a[0] + b[0], a[1] + b[1])
    
    @staticmethod
    def mul(a: Point, k: float) -> Point:
        return (a[0] * k, a[1] * k)
    
    @staticmethod
    def length(a: Point) -> float:
        return math.hypot(a[0], a[1])
    
    @staticmethod
    def normalize(a: Point) -> Point:
        l = GeometryUtils.length(a)
        if l == 0:
            return (0.0, 0.0)
        return (a[0] / l, a[1] / l)
    
    @staticmethod
    def rot(a: Point, angle_rad: float) -> Point:
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        return (a[0] * c - a[1] * s, a[0] * s + a[1] * c)
    
    @staticmethod
    def point_to_segment_distance(p: Point, a: Point, b: Point) -> float:
        ap = GeometryUtils.sub(p, a)
        ab = GeometryUtils.sub(b, a)
        ab2 = GeometryUtils.dot(ab, ab)
        if ab2 == 0:
            return GeometryUtils.length(ap)
        t = max(0.0, min(1.0, GeometryUtils.dot(ap, ab) / ab2))
        proj = GeometryUtils.add(a, GeometryUtils.mul(ab, t))
        return GeometryUtils.length(GeometryUtils.sub(p, proj))
    
    @staticmethod
    def rectangle_corners(center: Point, half_length: float, half_width: float, heading_rad: float) -> list[Point]:
        corners_local = [
            (half_length, half_width),
            (half_length, -half_width),
            (-half_length, -half_width),
            (-half_length, half_width),
        ]
        return [GeometryUtils.add(center, GeometryUtils.rot(c, heading_rad)) for c in corners_local]
    
    @staticmethod
    def project_polygon(vertices: list[Point], axis: Point) -> tuple[float, float]:
        ax = GeometryUtils.normalize(axis)
        dots = [GeometryUtils.dot(v, ax) for v in vertices]
        return min(dots), max(dots)
    
    @staticmethod
    def overlap_on_axis(av: list[Point], bv: list[Point], axis: Point) -> bool:
        a_min, a_max = GeometryUtils.project_polygon(av, axis)
        b_min, b_max = GeometryUtils.project_polygon(bv, axis)
        return not (a_max < b_min or b_max < a_min)


class Polygon:
    def __init__(self, vertices: list[Point]):
        self.vertices = vertices

    def edges(self) -> Iterable[tuple[Point, Point]]:
        n = len(self.vertices)
        for i in range(n):
            yield self.vertices[i], self.vertices[(i + 1) % n]

    def contains_point(self, p: Point) -> bool:
        # Бросание луча
        x, y = p
        inside = False
        v = self.vertices
        n = len(v)
        for i in range(n):
            x1, y1 = v[i]
            x2, y2 = v[(i + 1) % n]
            if ((y1 > y) != (y2 > y)):
                xin = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
                if x < xin:
                    inside = not inside
        return inside

    def distance_to_point(self, p: Point) -> float:
        # Расстояние до ребер многоугольника; отрицательное, если внутри
        if self.contains_point(p):
            return -min(GeometryUtils.point_to_segment_distance(p, a, b) for a, b in self.edges())
        return min(GeometryUtils.point_to_segment_distance(p, a, b) for a, b in self.edges())
    
    def intersects(self, other: "Polygon") -> bool:
        """Проверяет пересечение с другим многоугольником (SAT для выпуклых многоугольников)"""
        for poly in (self, other):
            vs = poly.vertices
            n = len(vs)
            for i in range(n):
                p1 = vs[i]
                p2 = vs[(i + 1) % n]
                axis = (-(p2[1] - p1[1]), p2[0] - p1[0])
                if not GeometryUtils.overlap_on_axis(self.vertices, other.vertices, axis):
                    return False
        return True
