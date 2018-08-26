__author__ = "Christian Kongsgaard"
__license__ = "MIT"

# -------------------------------------------------------------------------------------------------------------------- #
# Imports

# Module imports

import shapely
import shapely.geometry
import shapefile
import numpy as np
import typing


# Livestock imports


# -------------------------------------------------------------------------------------------------------------------- #
# Livestock Geometry Functions


def ray_triangle_intersection(ray_near, ray_dir, V):
    # Möller–Trumbore intersection algorithm in pure python
    # Based on http://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm

    v1 = V[0]
    v2 = V[1]
    v3 = V[2]
    eps = 0.000001
    edge1 = v2 - v1
    edge2 = v3 - v1
    pvec = np.cross(ray_dir, edge2)
    det = edge1.dot(pvec)

    if abs(det) < eps:
        return False, None

    inv_det = 1. / det
    tvec = ray_near - v1
    u = tvec.dot(pvec) * inv_det
    if u < 0. or u > 1.:
        return False, None

    qvec = np.cross(tvec, edge1)
    v = ray_dir.dot(qvec) * inv_det
    if v < 0. or u + v > 1.:
        return False, None

    t = edge2.dot(qvec) * inv_det
    if t < eps:
        return False, None

    return True, t


def lowest_face_vertex(v0, v1, v2):
    V = [v0, v1, v2]
    x0 = v0[0]
    y0 = v0[1]
    z0 = v0[2]
    x1 = v1[0]
    y1 = v1[1]
    z1 = v1[2]
    x2 = v2[0]
    y2 = v2[1]
    z2 = v2[2]
    X = [x0, x1, x2]
    Y = [y0, y1, y2]
    Z = [z0, z1, z2]

    Zsort = sorted(Z)

    if Zsort[0] == Zsort[2]:
        return np.array([sum(X) / 3, sum(Y) / 3, sum(Z) / 3])

    elif Zsort[0] < Zsort[1]:
        i = Z.index(Zsort[0])
        return V[i]

    elif Zsort[0] == Zsort[1]:
        i0 = Z.index(Zsort[0])
        i1 = Z.index(Zsort[1])
        x = 0.5 * (X[i0] + X[i1])
        y = 0.5 * (Y[i0] + Y[i1])
        return np.array([x, y, Zsort[0]])

    else:
        print('Error finding lowest point!')
        print('v0:', v0)
        print('v1:', v1)
        print('v2:', v2)
        return None


def angle_between_vectors(v1, v2, force_angle=None):
    # Computes the angle between two vectors.
    # :param v1: Vector1 as numpy array
    # :param v2: Vector2 as numpy array
    # :param force_angle: Default is None. Use to force angle into acute or obtuse.
    # :return: Angle in radians and its angle type.

    # Dot product
    dot_v1v2 = np.dot(v1, v2)

    # Determine angle type
    if dot_v1v2 > 0:
        angle_type = 'acute'

    elif dot_v1v2 == 0:
        return np.pi / 2, 'perpendicular'

    else:
        angle_type = 'obtuse'

    # Vector magnitudes and compute angle
    mag_v1 = np.sqrt(v1.dot(v1))
    mag_v2 = np.sqrt(v2.dot(v2))
    angle = np.arccos(abs(dot_v1v2 / (mag_v1 * mag_v2)))

    # Compute desired angle type
    if not force_angle:
        return angle, angle_type

    elif force_angle == 'acute':
        if angle_type == 'acute':
            return angle, 'acute'
        else:
            angle = np.pi - angle
            return angle, 'acute'

    elif force_angle == 'obtuse':
        if angle > np.pi / 2:
            return angle, 'obtuse'
        else:
            angle = np.pi - angle
            return angle, 'obtuse'
    else:
        print('force_angle has to be defined as None, acute or obtuse. '
              'force_angle was:', str(force_angle))
        return None, None


def line_intersection(p1, p2, p3, p4):
    # """
    # Computes the intersection between two lines given 4 points on those lines.
    # :param p1: Numpy array. First point on line 1
    # :param p2: Numpy array. Second point on line 1
    # :param p3: Numpy array. First point on line 2
    # :param p4: Numpy array. Second point on line 2
    # :return: Numpy array. Intersection point
    # """

    # Direction vectors
    v1 = (p2 - p1)
    v2 = (p4 - p3)

    # Cross-products and vector norm
    cv12 = np.cross(v1, v2)
    cpv = np.cross((p1 - p3), v2)
    t = np.linalg.norm(cpv) / np.linalg.norm(cv12)

    return p1 + t * v1


def obj_to_lists(obj_file: str) -> tuple:
    """
    Converts an .obj file into lists.

    :param obj_file: .obj file path
    :return: tuple with vertices, normals, faces
    """

    # Initialization
    vertices = []
    normals = []
    faces = []
    file = open(obj_file, 'r')
    lines = file.readlines()
    file.close()

    # Find data
    for line in lines:
        if line.startswith('v '):
            data = line.split(' ')
            vertices.append((float(data[1]),
                             float(data[2]),
                             float(data[3].strip())))

        elif line.startswith('vn'):
            data = line.split(' ')
            normals.append((float(data[1]),
                            float(data[2]),
                            float(data[3].strip())))

        elif line.startswith('f'):
            data = line.split(' ')
            d = []
            for elem in data[1:]:
                d0 = []
                for e in elem.strip().split('/'):
                    try:
                        d0.append(int(e))
                    except ValueError:
                        d0.append(None)
                d.append(d0)

            faces.append(d)

    return vertices, normals, faces


def centroid_z(polygon: shapely.geometry.Polygon) -> float:
    """
    Calculates the mean z-value from a Shapely polygon.

    :param polygon: Shapely Polygon with z-values.
    :type polygon: shapely.geometry.Polygon
    :return: Mean z-value of the polygon
    :rtype: float
    """

    z_values = []
    for pt in polygon.exterior.coords:
        z_values.append(pt[2])

    mean_z = sum(z_values) / len(z_values)

    return mean_z


def obj_to_polygons(obj_file: str) -> typing.List[shapely.geometry.Polygon]:
    """
    Converts an .obj file into a list of shapely polygons.

    :param obj_file: .obj file path
    :type obj_file: str
    :return: Shapely polygons in a list
    :rtype: list
    """

    vertices, normals, faces = obj_to_lists(obj_file)

    polygons = []
    for face in faces:
        face_vertices = []
        for vertex, _, __ in face:
            face_vertices.append(vertices[vertex - 1])

        polygon = shapely.geometry.Polygon(face_vertices)
        polygons.append(polygon)

    return polygons


def shapely_to_pyshp(shapely_geometry: shapely.geometry.Polygon) \
        -> shapefile._Shape:
    """
    This function converts a shapely geometry into a geojson and then into a
    pyshp object. Copied from Karim Bahgat's answer at:
    https://gis.stackexchange.com/questions/52705/how-to-write-shapely-geometries-to-shapefiles

    :param shapely_geometry: Shapely geometry to convert.
    :return: pyshp record object
    """

    geo_json = shapely.geometry.mapping(shapely_geometry)

    # create empty pyshp shape
    record = shapefile._Shape()

    # set shape type
    if geo_json["type"] == "Null":
        pyshp_type = 0
    elif geo_json["type"] == "Point":
        pyshp_type = 1
    elif geo_json["type"] in ["LineString", "MultiLineString"]:
        pyshp_type = 3
    elif geo_json["type"] == "MultiPolygon":
        pyshp_type = 5
    elif geo_json["type"] == "MultiPoint":
        pyshp_type = 8
    elif geo_json["type"] == "Polygon":
        pyshp_type = 15
    else:
        pyshp_type = 0

    record.shapeType = pyshp_type

    # set points and parts
    if geo_json["type"] == "Point":
        record.points = geo_json["coordinates"]
        record.parts = [0]

    elif geo_json["type"] in ("MultiPoint", "Linestring"):
        record.points = geo_json["coordinates"]
        record.parts = [0]

    elif geo_json["type"] in "Polygon":
        record.points = geo_json["coordinates"][0]
        record.parts = [0]

    elif geo_json["type"] in ("MultiPolygon", "MultiLineString"):
        index = 0
        points = []
        parts = []
        for each_multi in geo_json["coordinates"]:
            points.extend(each_multi[0])
            parts.append(index)
            index += len(each_multi[0])

        record.points = points
        record.parts = parts

    return record


def obj_to_shp(obj_file: str, shp_file: str) -> None:
    """
    Convert an .obj file into a shape file.

    :param obj_file: Path to .obj file
    :param shp_file: File path for shapefile
    :return: None
    """

    polygons = obj_to_polygons(obj_file)

    shape_writer = shapefile.Writer()
    shape_writer.field('id', 'N')
    shape_writer.field('height', 'N', decimal=5)

    for index, polygon in enumerate(polygons):
        converted_shape = shapely_to_pyshp(polygon)
        shape_writer._shapes.append(converted_shape)
        shape_writer.record(index, centroid_z(polygon))

    shape_writer.save(shp_file)

    return None
