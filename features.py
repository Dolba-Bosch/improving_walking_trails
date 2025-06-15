import shapely
from shapely.ops import nearest_points
from shapely.geometry import Point
from shapely.strtree import STRtree
import psycopg2
from helper_functions import get_coordinates, distance
from shapely import get_coordinates, set_coordinates, transform, wkb
import numpy as np


# Database connection, psycopg2 gebruiken om de geom op te halen
conn = psycopg2.connect(
    dbname="groningen_data",
    user="postgres",
    password="",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()


# Query geometry 
forests_query = """
SELECT landuse, ST_Transform(way, 4326) as way, "natural"
FROM planet_osm_polygon
WHERE landuse IN ('forest') OR "natural" IN ('wood');
"""
cursor.execute(forests_query)
hex_strings = [row[1] for row in cursor.fetchall()]



geometries = []
for hex_str in hex_strings:
    # Convert HEX string to bytes
    wkb_bytes = bytes.fromhex(hex_str)
    
    # Parse with Shapely (handle EWKB with SRID)
    geom = wkb.loads(wkb_bytes, hex=False)
    coords_forests = get_coordinates(geom)
    coords_swapped = np.flip(coords_forests, axis=1)
    
    # Update geometry
    swapped_geom = set_coordinates(geom, coords_swapped)
    geometries.append(swapped_geom)


# Build STRtree
forests_tree = STRtree(geometries)

# getting features!
def forest_distance(graph, edge):
    # bereken de afstand van de edge tot de dichtstbijzijnde bos
    # pak de nodes van de edge
    edge = (int(edge[0]), int(edge[1]))
    start_node = graph.vs[edge[0]]["coords"]
    end_node = graph.vs[edge[1]]["coords"]
    
    # maak een lijn van de edge
    line = shapely.LineString([start_node, end_node])
    
    # zoek de dichtstbijzijnde bos
    nearest_forest = forests_tree.nearest(line)
    nearest_forest = forests_tree.geometries.take(nearest_forest)
    p1, p2 = nearest_points(line, nearest_forest)
    # bereken de afstand
    dist = distance(p1.y, p1.x, p2.y, p2.x)
    
    return dist

def nearest_forest_area(graph, edge):
    # bereken de oppervlakte van het dichtstbijzijnde bos
    # pak de nodes van de edge
    edge = (int(edge[0]), int(edge[1]))
    start_node = graph.vs[edge[0]]["coords"]
    end_node = graph.vs[edge[1]]["coords"]
    
    # maak een lijn van de edge
    line = shapely.LineString([start_node, end_node])
    
    # zoek de dichtstbijzijnde bos
    nearest_forest = forests_tree.nearest(line)
    nearest_forest = forests_tree.geometries.take(nearest_forest)
    # bereken de oppervlakte
    area = nearest_forest.area
    return area

def edge_length(graph, edge):
    # bereken de lengte van de edge
    # pak de nodes van de edge
    edge = (int(edge[0]), int(edge[1]))
    start_node = graph.vs[edge[0]]["coords"]
    end_node = graph.vs[edge[1]]["coords"]
    length = distance(start_node[0], start_node[1], end_node[0], end_node[1])
    
    return length

def road_type(graph, edge):
    # pak de nodes van de edge
    edge = (int(edge[0]), int(edge[1]))
    roadtype = graph.es.find(_source=edge[0], _target=edge[1])["roadtype"]
    return roadtype
