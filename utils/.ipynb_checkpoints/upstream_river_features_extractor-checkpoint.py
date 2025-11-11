import rasterio
import numpy as np
from rasterio import mask, windows, plot
from shapely import plotting, Point
import shapely as shp
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.notebook import tqdm


class Feature:
    def __init__(self, name, fun, params=None):
        self.name = name
        self.fun = fun
        self.params = params

    def apply(self, data):
        if self.params:
            return self.fun(data, **self.params)
        else:
            return self.fun(data)


class UpstreamRiverFeaturesExtractor:
    def __init__(
        self,
        *,
        gdf,
        exclude_overlap,
        plot_results,
        features,
        buffers,
    ):
        self.gdf = gdf
        self.exclude_overlap = exclude_overlap
        self.features = features
        self.buffers = buffers
        self.plot_results = plot_results

        if self.plot_results:
            cmap = mpl.colormaps["plasma"]
            n_distances = len(self.gdf["distance"].unique())
            n_buffers = len(self.buffers)
            self.colors = cmap(np.linspace(0, 1, n_distances * n_buffers))

    def fill_features_dict_with_nan(self, *, features_dict, buffer, distance):
        for feature in self.features:
            features_dict[f"{feature.name}_buffer{buffer}_distance{distance}"] = [
                np.nan
            ]

        return features_dict

    def features_from_extracted_data(
        self, *, features_dict, extracted_data, buffer, distance
    ):
        for feature in self.features:
            features_dict[f"{feature.name}_buffer{buffer}_distance{distance}"] = [
                feature.apply(extracted_data)
            ]

        return features_dict

    def extract_data_window(self, *, src, bbox):
        # Convert the geographic bbox to pixel coordinates
        window = windows.from_bounds(*bbox, transform=src.transform)
        window_transform = windows.transform(window, src.transform)

        # Read data from the specified window
        window_data = src.read(1, window=window)
        window_data[window_data == src.nodata] = 0

        return window_data, window_transform

    def plot_bounds_data(self, *, src, bbox, ax, cmap):
        window_data, window_transform = self.extract_data_window(src=src, bbox=bbox)
        plot.show(window_data, transform=window_transform, ax=ax, cmap=cmap)

    def nodata_from_type(self, dtype: str):
        if "int" in dtype:
            nodata = 0
        else:
            nodata = np.nan

        return nodata

    def filter_nan(self, data):
        return data[~np.isnan(data)]

    def extract_data_under_geometry(self, *, geom, src):
        nodata = self.nodata_from_type(src.dtypes[0])

        try:
            data, _ = mask.mask(
                src,
                [geom],
                crop=True,
                filled=False,
                nodata=nodata,
                all_touched=True,
                indexes=1,
            )
            data = data.astype(np.float32).filled(fill_value=np.nan)
        except ValueError as e:
            print(f"Error: {e}")
            print("Skipping this data...")
            data = np.array([np.nan])

        data = self.filter_nan(data)

        if len(data) == 0:
            data = [0]

        return data

    def features_data_under_geometry(
        self, *, geom, src, features_dict, buffer, distance
    ):
        river_data = self.extract_data_under_geometry(geom=geom, src=src)
        features_dict = self.features_from_extracted_data(
            features_dict=features_dict,
            extracted_data=river_data,
            buffer=buffer,
            distance=distance,
        )
        return features_dict

    def plot_buffer_polygon(self, *, geom, ax, color):
        plotting.plot_polygon(
            geom, ax=ax, add_points=False, alpha=0.4, color=color, edgecolor="black"
        )

    def geometry_difference(self, geom, prev_geom):
        geom_diff = geom.difference(prev_geom)

        # We only exlude the precedent geometry if this doesn't result in an empty geometry.
        # It means that in case the geometry would be empty, we take the previous distance/buffer size geometry.
        if not geom_diff.is_empty:
            geom = geom_diff
            prev_geom = prev_geom.union(geom)

        return geom, prev_geom

    def exclude_previous_buffer_from_geometry(self, *, geom, buffer):
        # We either do the difference between the previous distance shape with the minimum buffer
        # or the difference with the distance but previous buffer size.
        if buffer == 1:
            geom, self.prev_dist_geom = self.geometry_difference(
                geom, self.prev_dist_geom
            )
        else:
            geom, self.prev_geom = self.geometry_difference(geom, self.prev_geom)

        return geom

    def extract_features(self, *, features_dict, geom, src, buffer, distance):
        if geom.is_empty:
            features_dict = self.fill_features_dict_with_nan(
                features_dict=features_dict, buffer=buffer, distance=distance
            )
        else:
            features_dict = self.features_data_under_geometry(
                geom=geom,
                src=src,
                features_dict=features_dict,
                buffer=buffer,
                distance=distance,
            )

        return features_dict

    def extract_station_features(self, *, src, date, station_group_idx, station_group):
        if date:
            station_name, station_date = station_group_idx[0], date
            features_dict = {"station": [station_name], "date": [station_date]}
        else:
            station_name = station_group_idx
            features_dict = {"station": station_name}

        if self.plot_results:
            _, ax = plt.subplots(1, figsize=(6, 6))
            color_idx = 0

        if self.exclude_overlap:
            self.prev_dist_geom = Point()

        for distance_idx, distance_row in station_group.iterrows():
            distance = distance_row["distance"]

            if self.exclude_overlap:
                self.prev_geom = Point()

            for buffer_idx, buffer in enumerate(self.buffers):
                geom = distance_row.geometry.buffer(buffer, cap_style="round")

                if self.exclude_overlap:
                    geom = self.exclude_previous_buffer_from_geometry(
                        geom=geom, buffer=buffer
                    )

                features_dict = self.extract_features(
                    features_dict=features_dict,
                    geom=geom,
                    src=src,
                    buffer=buffer,
                    distance=distance,
                )

                if self.plot_results and not geom.is_empty:
                    self.plot_bounds_data(
                        src=src, bbox=shp.box(*geom.bounds).bounds, ax=ax, cmap="Greens"
                    )
                    self.plot_buffer_polygon(
                        geom=geom,
                        ax=ax,
                        color=self.colors[color_idx % len(self.colors)],
                    )

                    ax.axis("off")
                    ax.set_title(f"{station_name}")
                    color_idx += 1

        if self.plot_results:
            plt.show()
            plt.close()

        return pd.DataFrame(features_dict)

    def extract(self, *, raster_path, output_file=None, date=None, save_to_file=False):
        df_list = []

        with rasterio.open(raster_path) as src:
            groups = self.gdf.groupby(["station"])

            for station_group_idx, station_group in tqdm(groups):
                df = self.extract_station_features(
                    src=src,
                    date=date,
                    station_group_idx=station_group_idx,
                    station_group=station_group,
                )
                df_list.append(df)

        result_df = pd.concat(df_list)

        if save_to_file:
            result_df.to_parquet(output_file)

        return result_df
