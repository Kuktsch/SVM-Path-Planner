from collections.abc import Iterable
import math

Point = tuple[float, float]

class GeometryUtils:
    """Функции для работы с геометрией"""
    
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
    
    @staticmethod
    def _orient(a: Point, b: Point, c: Point) -> float:
        """Ориентация тройки точек (знак площади параллелограмма)."""
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    
    @staticmethod
    def _on_segment(a: Point, b: Point, p: Point, eps: float = 1e-12) -> bool:
        """Проверяет, что p лежит на отрезке [a,b] (с допуском eps)."""
        return (
            min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
            and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
            and abs(GeometryUtils._orient(a, b, p)) <= eps
        )
    
    @staticmethod
    def segments_intersect(a: Point, b: Point, c: Point, d: Point, eps: float = 1e-12) -> bool:
        """Проверяет пересечение отрезков [a,b] и [c,d]."""
        o1 = GeometryUtils._orient(a, b, c)
        o2 = GeometryUtils._orient(a, b, d)
        o3 = GeometryUtils._orient(c, d, a)
        o4 = GeometryUtils._orient(c, d, b)

        # Общий случай
        if (o1 > eps and o2 < -eps or o1 < -eps and o2 > eps) and (o3 > eps and o4 < -eps or o3 < -eps and o4 > eps):
            return True

        # Коллинеарные/граничные случаи
        if abs(o1) <= eps and GeometryUtils._on_segment(a, b, c, eps):
            return True
        if abs(o2) <= eps and GeometryUtils._on_segment(a, b, d, eps):
            return True
        if abs(o3) <= eps and GeometryUtils._on_segment(c, d, a, eps):
            return True
        if abs(o4) <= eps and GeometryUtils._on_segment(c, d, b, eps):
            return True

        return False
    
    @staticmethod
    def segment_intersects_polygon(a: Point, b: Point, poly: "Polygon") -> bool:
        """Проверяет, что отрезок [a,b] пересекает/касается многоугольника или лежит внутри."""
        if poly.contains_point(a) or poly.contains_point(b):
            return True
        for e1, e2 in poly.edges():
            if GeometryUtils.segments_intersect(a, b, e1, e2):
                return True
        return False
    
    @staticmethod
    def path_intersects_obstacles(path: list[Point], obstacles: list["Polygon"]) -> bool:
        """True, если путь пересекает хотя бы одно препятствие."""
        if len(path) < 2 or not obstacles:
            return False
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            for obs in obstacles:
                if GeometryUtils.segment_intersects_polygon(a, b, obs):
                    return True
        return False


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