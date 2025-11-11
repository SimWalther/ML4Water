import pyproj
import osmnx as ox
import matplotlib.pyplot as plt
import rasterio
import geopandas as gpd
import networkx as nx
import numpy as np
import momepy
import shapely as shp
from rasterio.transform import rowcol
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import nearest_points
from shapely import Point
from tqdm.notebook import tqdm


class RiverNetwork:
    def __init__(self):
        self.graph = None

    def process(self, *, lon, lat, upstream_distance, max_segment_size):
        pass

    def find_terminal_nodes(self, G):
        nodes = []

        for node in G.nodes():
            node_neighbours = list(G.neighbors(node))

            if len(node_neighbours) == 1:
                nodes.append(node)

        return nodes

    def find_river_split_nodes(self, G):
        nodes = []

        for node in G.nodes():
            node_neighbours = list(G.neighbors(node))

            if len(node_neighbours) > 2:
                nodes.append(node)

        return nodes

    def get_subpath(self, node_list, first_node, second_node):
        subpath = []
        add_nodes = False

        for node in node_list:
            if node == first_node:
                add_nodes = True
            if add_nodes:
                subpath.append(node)
            if node == second_node:
                break

        return subpath

    def remove_orientation(self, G, node_list, orientation):
        for first_node, second_node in zip(node_list, node_list[1:]):
            if orientation == "upstream":
                u, v = first_node, second_node
            else:
                u, v = second_node, first_node

            try:
                G.remove_edge(u, v)
            except:
                pass

        return G

    def remove_non_connected_components(self, G, node):
        station_connected_component = None

        for connected_component in nx.weakly_connected_components(G):
            # If the node is inside this connected component keep this connected component and skip the other ones
            if node in connected_component:
                station_connected_component = connected_component
                break

        if not station_connected_component:
            raise ValueError(
                f"Graph's closest node is not in a connected component: {node}!"
            )

        G = nx.DiGraph(G.subgraph(station_connected_component))
        return G

    def add_elevation(self, G, elevation_fun):
        nx.set_node_attributes(
            G,
            {node: elevation_fun(Point(node[0], node[1])) for node in G.nodes()},
            "elevation",
        )

        return G

    def remove_downstream_edges(self, G):
        terminal_nodes = self.find_terminal_nodes(G)
        split_nodes = self.find_river_split_nodes(G)
        subpath_list = []
        subpath_direction = []

        for terminal_node in terminal_nodes[1:]:
            shortest_paths = nx.all_shortest_paths(
                G, terminal_nodes[0], terminal_node, weight="length"
            )

            for path in shortest_paths:
                inflection_nodes = [
                    node for node in path if node in terminal_nodes + split_nodes
                ]

                for inflection_first, inflection_second in zip(
                    inflection_nodes, inflection_nodes[1:]
                ):
                    subpath = self.get_subpath(
                        path, inflection_first, inflection_second
                    )
                    subpath_elevations = [
                        G.nodes[node]["elevation"] for node in subpath
                    ]

                    # Split the subpath into two halves
                    half = len(subpath) // 2

                    # Compare the average altitude between the first half of the subpath and the second half.
                    # The reason why we do the mean of the two halves instead of
                    # only the differene between the first and second inflection
                    # is that we have low resolution altitude images. So, the
                    # altitude is not robust and averaging can smooth out a bit this effect.
                    mean_first = np.mean(subpath_elevations[:half])
                    mean_second = np.mean(subpath_elevations[half:])

                    # Add the subpath with its correct direction
                    subpath_list.append(subpath)

                    # We compare the average altitudes and add the correct direction.
                    # Note: if the altitudes are equal, we say its upstream.
                    if mean_first >= mean_second:
                        subpath_direction.append("upstream")
                    else:
                        subpath_direction.append("downstream")

        for subpath, direction in zip(subpath_list, subpath_direction):
            G = self.remove_orientation(G, subpath, orientation=direction)

        return G

    def remove_non_reachable_nodes(self, G, node, distance):
        distances, _ = nx.single_source_dijkstra(
            G, node, cutoff=distance, weight="length"
        )
        reachable_nodes = [n for n, dist in distances.items()]

        G = G.subgraph(reachable_nodes)

        return G

    def graph_from_line_strings(self, line_strings):
        G = momepy.gdf_to_nx(
            gpd.GeoDataFrame(
                {
                    "length": [line.length for line in line_strings],
                    "geometry": line_strings,
                }
            ),
            approach="primal",
            length="length",
        ).to_directed()

        return G

    def nearest_node(self, G, point):
        min_dist = np.inf
        closest_node = None

        for node in G.nodes(data=True):
            node_point = Point(node[1]["x"], node[1]["y"])
            current_distance = shp.distance(node_point, point)

            if current_distance < min_dist:
                min_dist = current_distance
                closest_node = node

        closest_node_name = closest_node[
            0
        ]  # Keep only the name, don't return node's data
        return closest_node_name, min_dist

    def line_strings_from_graph(self, G, from_crs=None, to_crs=None):
        edges = list(G.edges())

        transform_crs = from_crs and to_crs
        if transform_crs:
            transformer = pyproj.Transformer.from_crs(from_crs, to_crs, always_xy=True)

        line_strings = []
        for u, v in edges:
            start_x, start_y = G.nodes[u]["x"], G.nodes[u]["y"]
            end_x, end_y = G.nodes[v]["x"], G.nodes[v]["y"]

            if transform_crs:
                start_x, start_y = transformer.transform(start_x, start_y)
                end_x, end_y = transformer.transform(end_x, end_y)

            start_point = Point(start_x, start_y)
            end_point = Point(end_x, end_y)

            # Create line from start and end points
            line = LineString([start_point, end_point])
            line_strings.append(line)

        return line_strings

    def segmentized_line_strings_from_graph(
        self, G, max_segment_size, from_crs="EPSG:4326", to_crs="LV95"
    ):
        line_strings = []
        for line in self.line_strings_from_graph(G, from_crs=from_crs, to_crs=to_crs):
            line = line.segmentize(
                max_segment_length=max_segment_size
            )  # If a segment is more than 5 meters split it

            for start, end in zip(line.coords, line.coords[1:]):
                line_strings.append(LineString([start, end]))

        return line_strings


class RiverNetworkShapefile(RiverNetwork):
    def __init__(self, path):
        self.geometries = gpd.read_file(path).geometry.to_crs("LV95")

    def process(
        self, *, lon, lat, upstream_distance, min_distance_to_fetch, max_segment_size
    ):
        geometries = self.geometries.clip(
            shp.buffer(Point(lon, lat), max(upstream_distance, min_distance_to_fetch))
        )

        # Create a list of LineStrings for each edge
        # The idea is that we will have a graph with much more nodes so when searching for a node if the node is along the river
        # we will have at most 5 meters of difference between the point we searched and this node.
        # Otherwise the closest point could be something like 100 meters away and we would have problems arising from this.
        # Also, when we want point at a distance of 500 meters from instance, we will search along the nodes not the underlying geometries
        # so it ensure that the precision error is maximum 5 meters.
        line_strings = self.segmentized_line_strings_from_shapefile(
            geometries, max_segment_size
        )
        self.graph = self.graph_from_line_strings(line_strings)
        return self.nearest_node(self.graph, Point(lon, lat))

    def segmentized_line_strings_from_shapefile(self, geometries, max_segment_size):
        line_strings = []

        # We split all multi-part geometries into multiple single geometries.
        geometries = geometries.explode()

        for line in geometries:
            line = line.segmentize(max_segment_length=max_segment_size)

            for start, end in zip(line.coords, line.coords[1:]):
                line_strings.append(LineString([start, end]))

        return line_strings


class RiverNetworkOSM(RiverNetwork):
    def __init__(self):
        pass

    def process(
        self, *, lon, lat, upstream_distance, min_distance_to_fetch, max_segment_size
    ):
        lon_espg4326, lat_espg4326 = pyproj.Transformer.from_crs(
            "LV95", "EPSG:4326", always_xy=True
        ).transform(lon, lat)

        try:
            G = self.fetch_river_from_OSM(
                lon_espg4326, lat_espg4326, upstream_distance, min_distance_to_fetch
            )
        except ValueError:
            print(
                f"OSM: no river found inside OSM bbox at {(lon_espg4326, lat_espg4326)}."
            )
            return None, np.inf

        # Create a list of LineStrings for each edge
        # The idea is that we will have a graph with much more nodes so when searching for a node if the node is along the river
        # we will have at most 5 meters of difference between the point we searched and this node.
        # Otherwise the closest point could be something like 100 meters away and we would have problems arising from this.
        # Also, when we want point at a distance of 500 meters from instance, we will search along the nodes not the underlying geometries
        # so it ensure that the precision error is maximum 5 meters.
        line_strings = self.segmentized_line_strings_from_graph(
            G, max_segment_size, from_crs="EPSG:4326", to_crs="LV95"
        )
        self.graph = self.graph_from_line_strings(line_strings)
        return self.nearest_node(self.graph, Point(lon, lat))

    def fetch_river_from_OSM(self, lon, lat, upstream_distance, min_distance_to_fetch):
        G = ox.graph.graph_from_point(
            (lat, lon),
            dist=max(upstream_distance, min_distance_to_fetch),
            dist_type="bbox",
            network_type="all",
            simplify=False,
            retain_all=True,
            truncate_by_edge=True,
            custom_filter='["waterway"]',
        ).to_undirected()

        return G


class UpstreamRiverExtractor:
    def __init__(
        self,
        stations_df,
        upstream_distances,
        dem_raster_path,
        max_segment_size,
        min_distance_to_fetch=1000,
        river_networks={
            "ecomorphology": RiverNetworkShapefile(
                "/home/simon.walther/shared-projects/ML4WATER/RAW_DATA/shapefiles/ecomorphology/romandie/romandie.shp"
            ),
            "river_toplogy": RiverNetworkShapefile(
                "/home/simon.walther/shared-projects/ML4WATER/RAW_DATA/shapefiles/typology/romandie/romandie.shp"
            ),
            "openstreetmap": RiverNetworkOSM(),
        },
    ):
        self.stations_df = stations_df
        self.max_segment_size = max_segment_size
        self.min_distance_to_fetch = min_distance_to_fetch
        self.elevation_at_point = self.get_elevation_at_point(
            raster_path=dem_raster_path
        )
        self.upstream_distances = upstream_distances
        self.river_networks = river_networks

    def get_elevation_at_point(self, raster_path):
        src = rasterio.open(raster_path)
        raster_data = src.read(1)
        transformer = pyproj.Transformer.from_crs("LV95", src.crs, always_xy=True)

        def get_row_col_elevation(point):
            x, y = transformer.transform(point.x, point.y)
            row, col = rowcol(src.transform, x, y)
            return raster_data[row, col]

        return get_row_col_elevation

    def extract(self):
        stations_upstream_river = []
        stations_upstream_river_distance = []
        stations_upstream_river_code = []
        stations_upstream_river_provider = []
        stations_upstream_river_lon = []
        stations_upstream_river_lat = []

        for upstream_distance in tqdm(self.upstream_distances):
            for idx, station in tqdm(self.stations_df.iterrows(), leave=False):
                lon, lat = station.lv95_Longitude, station.lv95_Latitude

                river_network_closest = None
                river_provider_used = None
                min_distance = np.inf
                closest_node = None

                # Find the shapefile with the closest node
                for river_provider, river_network in self.river_networks.items():
                    graph_closest_node, graph_node_dist = river_network.process(
                        lon=lon,
                        lat=lat,
                        upstream_distance=upstream_distance,
                        min_distance_to_fetch=self.min_distance_to_fetch,
                        max_segment_size=self.max_segment_size,
                    )

                    if graph_node_dist < min_distance:
                        river_network_closest = river_network
                        min_distance = graph_node_dist
                        closest_node = graph_closest_node
                        river_provider_used = river_provider

                if min_distance > 10:
                    print(
                        f"Warning: {station.name} closest node is {np.round(min_distance, 2)}m away from river!"
                    )
                    print("This data will not be used (!)")
                    continue

                river_network = river_network_closest
                G = river_network.graph

                try:
                    G = river_network.remove_non_connected_components(G, closest_node)
                except ValueError as e:
                    print(f"Error: {e}")
                    print(
                        f"Skipping station {station.name} at distance {upstream_distance}"
                    )
                    continue

                G = river_network.add_elevation(G, self.elevation_at_point)
                G = river_network.remove_downstream_edges(G)
                G = river_network.remove_non_reachable_nodes(
                    G, closest_node, upstream_distance
                )

                line_strings = river_network.line_strings_from_graph(G)
                multilines = MultiLineString(line_strings)

                stations_upstream_river.append(multilines)
                stations_upstream_river_distance.append(upstream_distance)
                stations_upstream_river_code.append(station.name)
                stations_upstream_river_provider.append(river_provider_used)
                stations_upstream_river_lon.append(lon)
                stations_upstream_river_lat.append(lat)

        upstream_river_gdf = (
            gpd.GeoDataFrame(
                {
                    "station": stations_upstream_river_code,
                    "lon": stations_upstream_river_lon,
                    "lat": stations_upstream_river_lat,
                    "geometry": stations_upstream_river,
                    "distance": stations_upstream_river_distance,
                    "provider": stations_upstream_river_provider,
                }
            )
            .set_geometry("geometry")
            .set_crs("LV95")
        )

        return upstream_river_gdf
