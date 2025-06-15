import polars as pl
import connectorx as cx
from math import radians, cos, sqrt



def run_sql_query(query):
    # functie wrapper voor het ophalen van data uit de database
    conn_str = "postgresql://postgres:@localhost:5432/groningen_data"
    data = pl.from_pandas(cx.read_sql(conn_str, query))
    return data


def distance(lat1, lon1, lat2, lon2):
    # De aarde straal in meters
    R = 6371000
    # Van graden naar radialen
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    x = (lon2 - lon1) * cos((lat1 + lat2) / 2)
    y = lat2 - lat1
    return R * sqrt(x**2 + y**2)


def get_coordinates(graph, route):
    # haal alle coordinaten op van de nodes in de route
    # dan kan je de route plotten met folium
    return [graph.vs[vertex]["coords"] for vertex in route]
