import gpxpy
import folium
import igraph as ig
import polars as pl
import connectorx as cx
import folium
from math import radians, cos, sqrt
import requests
import numpy as np
import json
import psycopg2
import logging
from shapely import get_coordinates, set_coordinates, transform, wkb
from shapely.geometry import Point, box
from shapely.ops import nearest_points
from shapely.strtree import STRtree
import shapely
from scipy import stats
from helper_functions import run_sql_query, distance
from features import forest_distance, nearest_forest_area, edge_length, road_type


# query om de wegen, nodes, coordinaten en tags op te halen
query = """SELECT
    w.id AS road_id,
    array_agg(n.id ORDER BY u.ordinality) AS node_ids,
    array_agg(n.lat / 10^7) AS node_lats,
    array_agg(n.lon / 10^7) AS node_lons,
    w.tags->>'highway' AS roadtype
    FROM planet_osm_ways AS w
    JOIN LATERAL unnest(w.nodes) WITH ORDINALITY AS u(node_id, ordinality)
    ON true
    JOIN planet_osm_nodes AS n
    ON n.id = u.node_id
    WHERE w.tags->>'highway' IS NOT NULL
    GROUP BY w.id;"""

data = run_sql_query(query)

print(data)


def create_graph(data):
    graph = ig.Graph(directed=False)
    
    node_coords = pl.DataFrame({
        "node_ids": data["node_ids"].explode(),
        "node_lats": data["node_lats"].explode(),
        "node_lons": data["node_lons"].explode()
    }).unique(subset=["node_ids"])
   
    coords = list(zip(
        node_coords["node_lats"].to_numpy(),
        node_coords["node_lons"].to_numpy()
    ))
    # voeg alle nodes toe aan de graph met de coordinaten
    graph.add_vertices(node_coords["node_ids"].cast(str).to_list())
    graph.vs["coords"] = coords
    
    # de existing nodes set gebruik ik bij het toevoegen van edges zodat ik alleen edges toevoeg tussen nodes die al in de graph zitten
    # zonder dit zou ik ook edges toevoegen tussen nodes die niet in de graph zitten.
    existing_nodes = set(node_coords["node_ids"].to_numpy())


    valid_edges = []
    roadtypes = []
    weights = []
    for node_ids_idx in range(len(data["node_ids"])):
        # zet alle node ids om naar strings
        str_ids = [str(id) for id in data["node_ids"][node_ids_idx]]
        
        for i in range(len(str_ids) - 1):
            # check of beide nodes al in de graph zitten
            if data["node_ids"][node_ids_idx][i] in existing_nodes and data["node_ids"][node_ids_idx][i+1] in existing_nodes:
                # voeg edge toe met de distance tussen de nodes als weight
                valid_edges.append((str_ids[i], str_ids[i+1]))
                roadtypes.append(data["roadtype"][node_ids_idx])
                weights.append(distance(
                    coords[i][0],
                    coords[i][1],
                    coords[i+1][0],
                    coords[i+1][1]
                ))

    # voeg alle edges en weights in een keer toe aan de graph
    graph.add_edges(valid_edges)
    graph.es["weight"] = weights
    graph.es["roadtype"] = roadtypes
    
    # neem de groote component van de graph
    if len(graph.components()) > 1:
        graph = graph.subgraph(max(graph.components(), key=len))
    
    return graph


graph = create_graph(data)

# zet node coords zodat hij correspondeert met de node ids die daadwerkelijk in de graph zitten
node_coords = pl.DataFrame({
    "node_ids": graph.vs["name"],
    "node_lats": [graph.vs[i]["coords"][0] for i in range(len(graph.vs))],
    "node_lons": [graph.vs[i]["coords"][1] for i in range(len(graph.vs))]
})

coords = list(zip(
    node_coords["node_lats"].to_numpy(),
    node_coords["node_lons"].to_numpy()
))

def get_node_coordinates(graph, route):
    # haal alle coordinaten op van de nodes in de route
    # dan kan je de route plotten met folium
    route = [int(vertex) for vertex in route]
    route = [graph.vs[vertex]["name"] for vertex in route]

    coordinates = []
    for i in range(len(route)):
        node_id = route[i]
        try:
            coords = graph.vs.find(name=str(node_id))["coords"]
            coordinates.append((coords[0], coords[1]))
        except Exception as e:
            print(f"Error: {e}")
            continue

    return coordinates


all_edges = graph.get_edgelist()
p1s = [Point(graph.vs[edge[0]]["coords"]) for edge in all_edges]
p2s = [Point(graph.vs[edge[1]]["coords"]) for edge in all_edges]
all_edges_lines = [shapely.LineString([p1, p2]) for p1, p2 in zip(p1s, p2s)]


road_network_tree = STRtree([Point(coord) for coord in coords])
road_network_edges_tree = STRtree(all_edges_lines)



pieterpad_points = []

all_files = [
    'pieterpad/01_pieterburen_winsum.gpx',
    'pieterpad/02_winsum_groningen_incl_omleiding.gpx',
    'pieterpad/03_groningen_zuidlaren.gpx',
    'pieterpad/04_zuidlaren_rolde.gpx',
    'pieterpad/05_rolde_schoonloo.gpx',
    'pieterpad/06_schoonloo_sleen.gpx',
    'pieterpad/07_sleen_coevorden.gpx',
    'pieterpad/08_coevorden_hardenberg.gpx',
    'pieterpad/09_hardenberg_ommen.gpx',
    'pieterpad/10_ommen_hellendoorn.gpx',
    'pieterpad/11_hellendoorn_holten.gpx',
    'pieterpad/12_holten_laren.gpx',
    'pieterpad/13_laren_vorden.gpx',
    'pieterpad/14_vorden_zelhem.gpx',
    'pieterpad/15_zelhem_braamt.gpx',
    'pieterpad/16_braamt_millingen.gpx',
    'pieterpad/17_millingen_groesbeek.gpx',
    'pieterpad/18_groesbeek_gennep.gpx',
    'pieterpad/19_gennep_vierlingsbeek.gpx',
    'pieterpad/20_vierlingsbeek_swolgen.gpx',
    'pieterpad/21_swolgen_venlo.gpx',
    'pieterpad/22_venlo_swalmen.gpx',
    'pieterpad/23_swalmen_montfort.gpx',
    'pieterpad/24_montfort_sittard.gpx',
    'pieterpad/25_sittard_strabeek.gpx',
    'pieterpad/26_strabeek_maastricht.gpx',
    ]

#overijssel
subset = all_files[8:11]  

#limburg
#subset = all_files[21:26]

# drenthe
#subset = all_files[3:6]  

for file in subset:
    with open(file, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    pieterpad_points.append((point.latitude, point.longitude))



# plot pieterpad
m = folium.Map(location=[53.2794, 6.5665], zoom_start=12)
for point in pieterpad_points:
    folium.Marker(location=point).add_to(m)
m.save("pieterpad.html")

osm_pieterpad_ids = []
for point in pieterpad_points:
    # pak de dichtstbijzijnde node bij de pieterpad punten
    closest_node = road_network_tree.nearest(Point(point))
    closest_node_point = road_network_tree.geometries.take(closest_node)

    dist = distance(point[0], point[1], closest_node_point.x, closest_node_point.y)
    # pak de node id van de dichtstbijzijnde node
    if dist < 3:
        osm_pieterpad_ids.append(closest_node)

# transformeren van tree-ids naar node-ids van OSM
osm_pieterpad_ids = [int(node) for node in osm_pieterpad_ids]
osm_pieterpad_ids = [node_coords["node_ids"][i] for i in osm_pieterpad_ids]


route_edges = []

def shortest_route_node_ids(graph, start_node, end_node):
    # haal de kortste route op tussen twee nodes
    start_idx = graph.vs.find(name=str(start_node)).index
    end_idx = graph.vs.find(name=str(end_node)).index
    path = graph.get_shortest_paths(start_idx, end_idx, weights="weight")
    return path[0]


for i in range(len(osm_pieterpad_ids) - 1):
    #get graph edges
    if not osm_pieterpad_ids[i] == osm_pieterpad_ids[i+1]:
        # pak kortste route tussen de twee pieterpad punten
        route_edges.append(shortest_route_node_ids(graph, osm_pieterpad_ids[i], osm_pieterpad_ids[i+1]))

#zet alle node ids en edges van het pieterpad in een lijst
route_nodes = [node for sublist in route_edges for node in sublist]
route_edges = [(route_nodes[i], route_nodes[i+1]) for i in range(len(route_nodes) - 1)]



    
# maak een dataset van de edges in het pieterpad
dataset = pl.DataFrame({
    "edge": route_edges,
    "forest_distance": [forest_distance(graph, edge) for edge in route_edges],
    "forest_area": [nearest_forest_area(graph, edge) for edge in route_edges],
    "roadtype": [road_type(graph, edge) for edge in route_edges],
    "edge_length": [edge_length(graph, edge) for edge in route_edges],
    "in_pieterpad": 1
})


# pak de edges in een 300 meter radius om de pieterpad punten
# omdat de shapely tree lat en lon gebruikt, moeten we zelf checken of de afstand tot de edge minder dan 300 meter is
set_of_edges_not_in_pieterpad = set()
for i in range(len(osm_pieterpad_ids)):

    # pak de coords van het pieterpad punt
    coords = graph.vs.find(name=str(osm_pieterpad_ids[i]))["coords"]

    # pak de edges in de radius van meer dan 300 meter
    nearby_roads = road_network_edges_tree.query(Point(coords).buffer(0.003))
    for edge_idx in nearby_roads:
        # check voor elke edge of de afstand tot het pieterpad punt minder dan 300 meter is
        edge = road_network_edges_tree.geometries.take(edge_idx)
        closest_edgepoint = nearest_points(Point(coords), edge)[1]

        dist= distance(coords[0], coords[1], closest_edgepoint.x, closest_edgepoint.y)
        if dist < 300:
            set_of_edges_not_in_pieterpad.add(all_edges[edge_idx])

# gooi de edges die niet in het pieterpad zitten in een dataset
dataset_not_in_pieterpad = pl.DataFrame({
    "edge": list(set_of_edges_not_in_pieterpad),
    "forest_distance": [forest_distance(graph, edge) for edge in set_of_edges_not_in_pieterpad],
    "forest_area": [nearest_forest_area(graph, edge) for edge in set_of_edges_not_in_pieterpad],
    "roadtype": [road_type(graph, edge) for edge in set_of_edges_not_in_pieterpad],
    "edge_length": [edge_length(graph, edge) for edge in set_of_edges_not_in_pieterpad],
    "in_pieterpad": 0
})

full_dataset = pl.concat([dataset, dataset_not_in_pieterpad])

start_nodes = [full_dataset["edge"][i][0] for i in range(len(full_dataset))]
end_nodes = [full_dataset["edge"][i][1] for i in range(len(full_dataset))]
# combineer de datasets van de start en end nodes
new_dataset = pl.DataFrame({
    "start_node": start_nodes,
    "end_node": end_nodes,
    "roadtype": full_dataset["roadtype"],
    "forest_distance": full_dataset["forest_distance"],
    "forest_area": full_dataset["forest_area"],
    "edge_length": full_dataset["edge_length"],
    "in_pieterpad": full_dataset["in_pieterpad"]
})

# verwijder dubbele entries in de dataset
new_dataset = new_dataset.unique(subset=["start_node", "end_node", "roadtype", "forest_distance","forest_area", "edge_length", "in_pieterpad"])
# verwijder edges waar de start en end node hetzelfde zijn
# dit kan soms voorkomen als de pieterpad punten heel dicht bij elkaar liggen
new_dataset = new_dataset.filter((pl.col("start_node") != pl.col("end_node")))

with open('dataset.csv', 'w') as f:
    f.write(new_dataset.write_csv())

# plot alle wegen in de dataset
def plot_all_roads(dataset):
    m = folium.Map(location=[53.2794, 6.5665], zoom_start=12)
    for i in range(len(dataset)):
        start_node = dataset["start_node"][i]
        end_node = dataset["end_node"][i]
        coords = [graph.vs[start_node]["coords"], graph.vs[end_node]["coords"]]
        if dataset["in_pieterpad"][i] == 1:
            folium.PolyLine(coords, color="red", weight=8.5, opacity=1, tooltip="In pieterpad: " + str(dataset["in_pieterpad"][i]) +
                     " Roadtype: " + str(dataset["roadtype"][i]) +
                     " Forest distance: " + str(dataset["forest_distance"][i]) +
                        "start end node: " + str(dataset["start_node"][i]) + " " + str(dataset["end_node"][i])
                    ).add_to(m)
        else:
            folium.PolyLine(coords, color="blue", weight=2.5, opacity=0.7, tooltip="In pieterpad: " + str(dataset["in_pieterpad"][i]) +
                     " Roadtype: " + str(dataset["roadtype"][i]) +
                     " Forest distance: " + str(dataset["forest_distance"][i]) +
                        "start end node: " + str(dataset["start_node"][i]) + " " + str(dataset["end_node"][i])
                    ).add_to(m)
    m.save("dataset.html")

plot_all_roads(new_dataset)


def distance_logit_model(forest_distance, forest_area, edge_length, roadtype):
# R output van het logit model:
#    Coefficients:
#                  Estimate Std. Error z value Pr(>|z|)    
#(Intercept)     -1.631e+00  1.151e-01 -14.164  < 2e-16 ***
#track            4.220e-01  9.353e-02   4.512 6.42e-06 ***
#path            -9.326e-01  1.024e-01  -9.107  < 2e-16 ***
#footway         -2.498e+00  3.033e-01  -8.238  < 2e-16 ***
#cycleway        -5.126e-01  1.125e-01  -4.558 5.17e-06 ***
#tertiary        -6.606e-01  1.855e-01  -3.562 0.000368 ***
#service         -2.089e+00  2.231e-01  -9.364  < 2e-16 ***
#primary         -3.793e+00  1.006e+00  -3.769 0.000164 ***
#residential     -9.230e-01  1.469e-01  -6.282 3.34e-10 ***
#trunk           -1.529e+01  3.604e+02  -0.042 0.966171    
#pedestrian      -1.511e+01  2.680e+02  -0.056 0.955057    
#secondary       -1.499e+01  3.814e+02  -0.039 0.968640    
#bridleway       -1.470e+01  7.584e+02  -0.019 0.984537    
#rest_area       -1.473e+01  8.468e+02  -0.017 0.986118    
#forest_area     -9.634e+04  1.176e+04  -8.189 2.63e-16 ***
#log_edge_length  1.151e-01  2.620e-02   4.393 1.12e-05 ***
    log_edge_length = np.log(edge_length)
    latent_variable = -1.631 + forest_area * -9.634e-04 + log_edge_length * 0.1151
    roadtype_penalty = {
            "track": 0.422,
            "path": -0.9326,
            "footway": -2.498,
            "cycleway": -0.5126,
            "tertiary": -0.6606,
            "service": -2.089,
            "primary": -3.793,
            "residential": -0.923,
            "trunk": -15.29,
            "motorway": -15.29,
            "pedestrian": -15.11,
            "secondary": -14.99,
            "bridleway": -14.70,
            "rest_area": -14.73
            }
    roadtype_penalty = roadtype_penalty.get(roadtype, 0)
    latent_variable += roadtype_penalty
    prob = stats.norm.cdf(latent_variable)
    return -np.log(prob) * edge_length



# graph maken met de logit model als weight
def create_graph_logit(data):
    graph = ig.Graph(directed=False)
    
    node_coords = pl.DataFrame({
        "node_ids": data["node_ids"].explode(),
        "node_lats": data["node_lats"].explode(),
        "node_lons": data["node_lons"].explode()
    }).unique(subset=["node_ids"])
   
    coords = list(zip(
        node_coords["node_lats"].to_numpy(),
        node_coords["node_lons"].to_numpy()
    ))
    # voeg alle nodes toe aan de graph met de coordinaten
    graph.add_vertices(node_coords["node_ids"].cast(str).to_list())
    graph.vs["coords"] = coords
    
    # de existing nodes set gebruik ik bij het toevoegen van edges zodat ik alleen edges toevoeg tussen nodes die al in de graph zitten
    # zonder dit zou ik ook edges toevoegen tussen nodes die niet in de graph zitten.
    existing_nodes = set(node_coords["node_ids"].to_numpy())
    valid_edges = []
    roadtypes = []
    for node_ids_idx in range(len(data["node_ids"])):
        # zet alle node ids om naar strings
        str_ids = [str(id) for id in data["node_ids"][node_ids_idx]]
        
        for i in range(len(str_ids) - 1):
            # check of beide nodes al in de graph zitten
            if data["node_ids"][node_ids_idx][i] in existing_nodes and data["node_ids"][node_ids_idx][i+1] in existing_nodes:
                # voeg edge toe met de distance tussen de nodes als weight
                valid_edges.append((str_ids[i], str_ids[i+1]))
                roadtypes.append(data["roadtype"][node_ids_idx])
    # voeg alle edges en weights in een keer toe aan de graph
    graph.add_edges(valid_edges)
    graph.es["roadtype"] = roadtypes
    weights = [distance_logit_model(forest_distance(graph, edge), nearest_forest_area(graph, edge), edge_length(graph, edge), road_type(graph,edge)) for edge in graph.get_edgelist()]
    graph.es["weight"] = weights
    # neem de groote component van de graph
    if len(graph.components()) > 1:
        graph = graph.subgraph(max(graph.components(), key=len))
    return graph

graph_logit = create_graph_logit(data)

def get_route(graph, start, end):
    start_vertex = graph.vs.find(name=str(start))
    end_vertex = graph.vs.find(name=str(end))
    return graph.get_shortest_paths(start_vertex, end_vertex, weights="weight", output="vpath")[0]




def address_to_node_id(graph, address):
    base_url = "https://nominatim.openstreetmap.org/search?q="
    address = address.replace(" ", "+")
    # add header to request
    headers = {
        'User-Agent': 'Improving_walking_trails'}
    response = requests.get(base_url + address + "&format=json", headers=headers)
    response = json.loads(response.text)
    lat = float(response[0]["lat"])
    lon = float(response[0]["lon"])
    closest_vertex = min(graph.vs, key=lambda vertex: distance(vertex["coords"][0], vertex["coords"][1], lat, lon))
    return closest_vertex["name"]

def routeplanner(graph, start_address, end_address):
    start = address_to_node_id(graph, start_address)
    end = address_to_node_id(graph, end_address)
    route = get_route(graph, start, end)
    return route

route = routeplanner(graph_logit, "Hardenberg, Overijssel", "Holten, Overijssel")
#route = routeplanner(graph_logit, "Stationsplein, Venlo", "Sint Pietersberg, Maastricht")
#route = routeplanner(graph_logit, "Zuidlaren, Drenthe", "Sleen, Drenthe")


# plot route met pieterpad
def plot_route_with_pieterpad(graph, route, pieterpad_points):
    m = folium.Map(location=[53.2794, 6.5665], zoom_start=12)
    coordinates = get_node_coordinates(graph, route)
    folium.PolyLine(coordinates, tooltip="Route", weight=5).add_to(m)
    
    # plot pieterpad points
    for point_idx in range(len(pieterpad_points)-1):
        if distance(
            pieterpad_points[point_idx][0],
            pieterpad_points[point_idx][1],
            pieterpad_points[point_idx+1][0],
            pieterpad_points[point_idx+1][1]
        ) < 2000:
            # plot the pieterpad points
            folium.PolyLine(
                [pieterpad_points[point_idx], pieterpad_points[point_idx + 1]],
                color="Red", weight=5, opacity=1, tooltip="Pieterpad"
            ).add_to(m)
    
    m.save("route_with_pieterpad.html")


plot_route_with_pieterpad(graph_logit, route, pieterpad_points)

