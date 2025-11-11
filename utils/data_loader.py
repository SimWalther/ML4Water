import os
import numpy as np
import pandas as pd
import geopandas as gpd
import configparser
from datetime import datetime
from water_temperature_loader import (
    BoironWaterTemperatureLoader,
    BroyeWaterTemperatureLoader,
    VenogeWaterTemperatureLoader,
    VaudWaterTemperatureLoader,
)
from air_temperature_loader import (
    BoironAirTemperatureLoader,
    BroyeAirTemperatureLoader,
    VenogeAirTemperatureLoader,
    VaudAirTemperatureLoader,
)
from meteo import SwissMeteoFeatures
from upstream_river_network_extractor import UpstreamRiverExtractor
from upstream_river_features_extractor import UpstreamRiverFeaturesExtractor, Feature

# TODO: move this in another file
mapping_dict_station_to_river = {
 'BOI011': 'Boiron',
 'BOI013': 'Boiron',
 'BOI020': 'Boiron',
 'RTV028': 'Boiron',
 'BOI018': 'Boiron',
 'BOI019': 'Boiron',
 'BOI015': 'Boiron',
 'D002': 'Dullive',
 'D001': 'Dullive',
 'RTV018': 'Dullive',
 'RTV063': 'Hongrin',
 'RTV062': 'Torneresse',
 'RTV003': 'Aubonne',
 'MRP001': 'Aubonne',
 'RTV072': 'Broye',
 'RTV015': 'Broye',
 'RTV022': 'Broye',
 'RTV070': 'Broye',
 'RTV046': 'Broye',
 'RTV074': 'Mentue',
 'RTV030': 'Mentue',
 'RTV037': 'Nozon',
 'RTV035': 'Nozon',
 'RTV036': 'Nozon',
 'RTV077': 'Orbe',
 'RTV076': 'Orbe',
 'RTV040': 'Orbe',
 'RTV092': 'Orbe',
 'RTV039': 'Orbe',
 'RTV055': 'Venoge',
 'RTV054': 'Venoge',
 'RTV053': 'Venoge',
 'RTV078': 'Venoge',
 'MRP005': 'Venoge',
 'RTV061': 'Veyron',
 'RTV060': 'Veyron',
 'RTV085': 'Veyron',
 'RTV024': 'Forestay',
 'RTV029': 'Lutrive',
 'RTV041': 'Paudèze',
 'RTV043': 'Promenthouse',
 'RTV047': 'Serine',
 'RTV050': 'Talent',
 'RTV052': 'Vaux',
 'RTV080': 'Versoix',
 'RTV071': 'Broye',
 'RTV014': 'Broye',
 'RTV069': 'Broye',
 'RTV068': 'Broye',
 'RTV073': 'Broye',
 'RTV067': 'Broye',
 'RTV066': 'Broye',
 'RTV065': 'Broye',
 'RTV013': 'Broye',
 'RTV064': 'Bioleyres',
 'BOI021': 'Boiron',
 'BOI012': 'Boiron',
 'RTV008': 'Boiron',
 'BOI009': 'Boiron',
 'RTV045': 'Boiron',
 'BOI006': 'Boiron',
 'BOI005': 'Boiron',
 'BOI001': 'Boiron',
 'RTV021': 'Boiron',
 'BOI017': 'Boiron',
 'BOI008': 'Boiron',
 'BOI016': 'Boiron'
}

class DataLoader:
    def __init__(
        self,
        config_path="config.ini",
        recreate=False,
    ):
        config = configparser.ConfigParser()
        config.read(config_path)

        # Options
        self.recreate = recreate

        # Sampling
        self.start_year = int(config["Sampling"]["start_year"])
        self.end_year = int(config["Sampling"]["end_year"])
        self.time_interval = config["Sampling"]["time_interval"]
        self.water_sampling_quantile = float(
            config["Sampling"]["water_sampling_quantile"]
        )
        self.air_sampling_quantile = float(config["Sampling"]["air_sampling_quantile"])
        self.min_temperature = float(config["Sampling"]["min_temperature"])
        self.max_temperature = float(config["Sampling"]["max_temperature"])
        self.ignore = config["Sampling"]["ignore"].split(", ")

        # Rasters
        self.rasters = config.items("Rasters")
        altitude_raster_name = config["RiverNetwork"]["altitude_raster"]
        self.upstream_altitude_path = f"{config['Rasters'][altitude_raster_name]}/{altitude_raster_name}.tif"

        # Upstream
        self.segments_length = {
            raster_name: int(config["Upstream"][f"{raster_name}_segments_length"])
            for raster_name, _ in self.rasters
        }

        self.max_distance = {
            raster_name: int(config["Upstream"][f"{raster_name}_max_distance"])
            for raster_name, _ in self.rasters
        }

        self.upstream_width = {
            raster_name: int(config["Upstream"][f"{raster_name}_width"])
            for raster_name, _ in self.rasters
        }

        self.upstream_max_width = {
            raster_name: int(config["Upstream"][f"{raster_name}_max_width"])
            for raster_name, _ in self.rasters
        }

        self.upstream_multiple_dates = {
            raster_name: config["Upstream"][f"{raster_name}_multiple_dates"] == "True"
            for raster_name, _ in self.rasters
        }

        self.upstream_feature = {
            raster_name: config["Upstream"][f"{raster_name}_feature"]
            for raster_name, _ in self.rasters
        }

        self.upstream_plot = {
            raster_name: config["Upstream"][f"{raster_name}_plot"] == "True"
            for raster_name, _ in self.rasters
        }

        # Paths
        self.data_dir = config["Paths"]["data_dir"]
        self.meteo_metadata = config["Paths"]["meteo_metadata"]
        self.meteo_dir = config["Paths"]["meteo_dir"]
        self.prepared_data_dir = config["Paths"]["prepared_data_dir"]
        self.boiron_water_temperature_path = config["Paths"][
            "boiron_water_temperature_path"
        ]
        self.broye_water_temperature_path = config["Paths"][
            "broye_water_temperature_path"
        ]
        self.venoge_water_temperature_path = config["Paths"][
            "venoge_water_temperature_path"
        ]
        self.vaud_water_temperature_path = config["Paths"][
            "vaud_water_temperature_path"
        ]
        self.boiron_air_temperature_path = config["Paths"][
            "boiron_air_temperature_path"
        ]
        self.broye_air_temperature_path = config["Paths"]["broye_air_temperature_path"]
        self.venoge_air_temperature_path = config["Paths"][
            "venoge_air_temperature_path"
        ]
        self.vaud_air_temperature_path = config["Paths"]["vaud_air_temperature_path"]

        # Upstream river network
        self.river_network_max_segment_size = float(
            config["RiverNetwork"]["max_segment_size"]
        )

    def trig_day_from_dayofyear(self, dayofyear):
        day_sin = (np.sin(2 * np.pi * dayofyear / 366) + 1) / 2
        day_cos = (np.cos(2 * np.pi * dayofyear / 366) + 1) / 2

        return day_cos, day_sin

    def trig_day(self, df):
        day_cos, day_sin = self.trig_day_from_dayofyear(
            df.index.get_level_values("date").dayofyear
        )
        return day_cos.to_list(), day_sin.to_list()

    def year_of_df(self, df):
        return df.index.get_level_values("date").year

    def merge_data(
        self, water_temperature_df, air_temperature_df, metadata_df, meteo_df, rasters_upstream_river_features
    ):
        stations_data = {}

        # Water data
        for station in water_temperature_df.columns:
            stations_data[station] = water_temperature_df[station]
            stations_data[station] = (
                stations_data[station]
                .reset_index()
                .rename({"Date et heure": "date", station: "water_max_temp"}, axis=1)
            )
            stations_data[station]["station"] = str(station)
            stations_data[station] = stations_data[station].set_index(
                ["date", "station"]
            )

        stations_data = pd.concat(stations_data.values())
        stations_data = stations_data.query(
            "water_max_temp >= @self.min_temperature and water_max_temp <= @self.max_temperature"
        )

        stations_data = stations_data.dropna(axis=0)

        # Air temperature data
        air_data = {}

        for air_station in air_temperature_df.columns:
            air_data[air_station] = air_temperature_df[air_station]
            air_data[air_station] = (
                air_data[air_station]
                .reset_index()
                .rename({air_station: "air_max_temp"}, axis=1)
            )
            air_data[air_station]["station"] = str(air_station)
            air_data[air_station] = air_data[air_station].set_index(["date", "station"])

        air_data = pd.concat(air_data.values())
        air_data = air_data.dropna(axis=0)
        air_data = air_data.groupby("date").mean()

        stations_data = stations_data.join(air_data)

        # Metadata
        stations_data = stations_data.join(metadata_df)

        # Station's river 
        stations_data = stations_data.reset_index()
        stations_data['river_name'] = stations_data['station'].map(mapping_dict_station_to_river)
        stations_data = stations_data.set_index(["date", "station"])

        # Day sin/cos
        day_cos, day_sin = self.trig_day(stations_data)
        stations_data["day_cos"] = day_cos
        stations_data["day_sin"] = day_sin
        stations_data["year"] = self.year_of_df(stations_data)

        # Meteo
        meteo_df = meteo_df.reset_index().rename({"time": "date"}, axis=1)
        meteo_df = meteo_df.pivot_table(
            index=["date", "station"], columns="feature", values="interpolated"
        )
        meteo_df = pd.DataFrame(
            meteo_df.to_records()
        )  # pivot table to regular dataframe
        meteo_df["date"] = pd.to_datetime(meteo_df["date"])
        meteo_df = meteo_df.set_index(["date", "station"])
        stations_data = stations_data.join(meteo_df)

        # Add altitude from rasters to stations data
        stations_altitudes = rasters_upstream_river_features["dhm25"].set_index('station') # TODO: take the key name from config
        altitudes_at_stations = stations_altitudes[stations_altitudes.columns[0]]
        altitudes_difference = altitudes_at_stations - stations_altitudes[stations_altitudes.columns[-1]]

        stations_data = stations_data.join(
            pd.DataFrame({
                'altitude': altitudes_at_stations,
                'altitude_difference': altitudes_difference,
            })
        )

        stations_data = stations_data.reset_index()

        # Make sure data are within dates
        stations_data = stations_data.query(
            "date.dt.year >= @self.start_year and date.dt.year <= @self.end_year"
        )

        # Remove stations to ignore
        stations_data = stations_data.query("station not in @self.ignore")

        return stations_data

    def load(self):
        water_temperature_df = self.load_water_temperature_data()
        air_temperature_df = self.load_air_temperature_data()
        meteo_features_df = self.load_meteo_features()
        metadata_df = self.load_stations_metadata()
        rasters_upstream_river_features = self.load_upstream_river_features()

        stations_data = self.merge_data(
            water_temperature_df, air_temperature_df, metadata_df, meteo_features_df, rasters_upstream_river_features
        )

        return {
            "stations_data": stations_data,
            "upstream_river_features": rasters_upstream_river_features,
        }

    def _load_upstream_river_network(self, segments_length, max_distance):
        print("Loading upstream river network...")
        file_path = f"{self.prepared_data_dir}/upstream_river_gdf_{segments_length}_to_{max_distance}.parquet"

        if self.recreate or not os.path.isfile(file_path):
            stations = self.load_stations_metadata()
            segments = np.arange(
                segments_length, max_distance + segments_length, segments_length
            )
            upstream_river_extractor = UpstreamRiverExtractor(
                stations,
                upstream_distances=segments,
                dem_raster_path=self.upstream_altitude_path,
                max_segment_size=self.river_network_max_segment_size,
            )
            result_df = upstream_river_extractor.extract()
            result_df.to_parquet(file_path)

        else:
            result_df = gpd.read_parquet(file_path)

        return result_df

    def extract_upstream_river_features(
        self,
        features,
        upstream_river_gdf,
        raster_name,
        file_path,
        full_raster_path,
        date,
    ):
        if self.recreate or not os.path.isfile(file_path):
            buffers = np.arange(
                self.upstream_width[raster_name],
                self.upstream_max_width[raster_name] + self.upstream_width[raster_name],
                self.upstream_width[raster_name],
            )

            features_extractor = UpstreamRiverFeaturesExtractor(
                gdf=upstream_river_gdf,
                exclude_overlap=True,
                plot_results=self.upstream_plot[raster_name],
                features=features,
                buffers=buffers,
            )

            return features_extractor.extract(
                raster_path=full_raster_path,
                output_file=file_path,
                date=date,
                save_to_file=True,
            )
        else:
            return pd.read_parquet(file_path)

    def load_upstream_river_features(self):
        rasters_upstream_river_features = dict()

        for raster_name, raster_path in self.rasters:
            features = []

            if self.upstream_feature[raster_name] == "mean":
                features.append(Feature("mean", np.nanmean))
            elif self.upstream_feature[raster_name] == "median":
                features.append(Feature("median", np.nanmedian))
            elif self.upstream_feature[raster_name] == "min":
                features.append(Feature("min", np.nanmin))

            upstream_river_gdf = self._load_upstream_river_network(
                self.segments_length[raster_name],
                self.max_distance[raster_name],
            )

            upstream_river_features = []

            if self.upstream_multiple_dates[raster_name]:
                for year in np.arange(self.start_year, self.end_year + 1):
                    print(f"Extract {raster_name} upstream features - {year}...")

                    date = datetime(year, 1, 1)
                    full_raster_path = f"{raster_path}/{raster_name}_{year}.tif"
                    file_path = f"{self.prepared_data_dir}/{raster_name}_{year}.parquet"

                    upstream_river_features.append(
                        self.extract_upstream_river_features(
                            features,
                            upstream_river_gdf,
                            raster_name,
                            file_path,
                            full_raster_path,
                            date,
                        )
                    )
            else:
                print(f"Extract {raster_name} upstream features...")

                date = None
                full_raster_path = f"{raster_path}/{raster_name}.tif"
                file_path = f"{self.prepared_data_dir}/{raster_name}.parquet"

                upstream_river_features.append(
                    self.extract_upstream_river_features(
                        features,
                        upstream_river_gdf,
                        raster_name,
                        file_path,
                        full_raster_path,
                        date,
                    )
                )

            rasters_upstream_river_features[raster_name] = pd.concat(
                upstream_river_features, axis=0
            ).query("station not in @self.ignore")

        return rasters_upstream_river_features

    def load_water_temperature_data(self):
        print("Loading water temperature data...")

        file_path = f"{self.prepared_data_dir}/water_temperature.parquet"
        if self.recreate or not os.path.isfile(file_path):
            boiron_water_temperature_data = BoironWaterTemperatureLoader(
                self.boiron_water_temperature_path,
                self.time_interval,
                self.water_sampling_quantile,
            ).load()

            broye_water_temperature_data = BroyeWaterTemperatureLoader(
                self.broye_water_temperature_path,
                self.time_interval,
                self.water_sampling_quantile,
            ).load()

            venoge_water_temperature_data = VenogeWaterTemperatureLoader(
                self.venoge_water_temperature_path,
                self.time_interval,
                self.water_sampling_quantile,
            ).load()

            df_columns = (
                boiron_water_temperature_data.columns.to_list()
                + broye_water_temperature_data.columns.to_list()
                + venoge_water_temperature_data.columns.to_list()
            )

            vaud_water_temperature_data = VaudWaterTemperatureLoader(
                self.vaud_water_temperature_path,
                self.time_interval,
                self.water_sampling_quantile,
                other_df_columns=df_columns,
            ).load()

            result_df = pd.concat(
                [
                    boiron_water_temperature_data,
                    broye_water_temperature_data,
                    venoge_water_temperature_data,
                    vaud_water_temperature_data,
                ],
                axis=1,
            )

            print("Saving water temperature data...")
            result_df.to_parquet(file_path)
        else:
            result_df = pd.read_parquet(file_path)

        return result_df

    def load_air_temperature_data(self):
        print("Loading air temperature data...")

        file_path = f"{self.prepared_data_dir}/air_temperature.parquet"
        if self.recreate or not os.path.isfile(file_path):
            boiron_air_temperature_data = BoironAirTemperatureLoader(
                self.boiron_air_temperature_path,
                self.time_interval,
                self.air_sampling_quantile,
            ).load()

            broye_air_temperature_data = BroyeAirTemperatureLoader(
                self.broye_air_temperature_path,
                self.time_interval,
                self.air_sampling_quantile,
            ).load()

            venoge_air_temperature_data = VenogeAirTemperatureLoader(
                self.venoge_air_temperature_path,
                self.time_interval,
                self.air_sampling_quantile,
            ).load()

            df_columns = (
                boiron_air_temperature_data.columns.to_list()
                + broye_air_temperature_data.columns.to_list()
                + venoge_air_temperature_data.columns.to_list()
            )

            vaud_air_temperature_data = VaudAirTemperatureLoader(
                self.vaud_air_temperature_path,
                self.time_interval,
                self.air_sampling_quantile,
                other_df_columns=df_columns,
            ).load()

            result_df = pd.concat(
                [
                    boiron_air_temperature_data,
                    broye_air_temperature_data,
                    venoge_air_temperature_data,
                    vaud_air_temperature_data,
                ],
                axis=1,
            )

            print("Saving air temperature data...")
            result_df.to_parquet(file_path)
        else:
            result_df = pd.read_parquet(file_path)

        return result_df

    def load_meteo_features(self):
        print("Loading and interpolating meteo features...")

        file_path = f"{self.prepared_data_dir}/meteo_features.parquet"
        if self.recreate or not os.path.isfile(file_path):
            swiss_meteo_features = SwissMeteoFeatures(
                self.load_meteo_stations_metadata(),
                self.load_stations_metadata(),
                self.meteo_dir,
            )
            result_df = swiss_meteo_features.get_features_at_stations()

            print("Saving meteo features...")
            result_df.to_parquet(file_path)
        else:
            result_df = pd.read_parquet(file_path)

        return result_df

    def load_stations_metadata(self):
        metadata_df = pd.read_parquet(f"{self.data_dir}/all_stations.parquet")

        # TODO: Filter ignored stations

        return (
            metadata_df.rename({"CODE_MdlR": "station", "Date": "date"}, axis=1)
            .groupby("station")
            .first()
            .drop(["Remark", "date"], axis=1)
        )

    def load_meteo_stations_metadata(self):
        return pd.read_csv(self.meteo_metadata).set_index("Station")
