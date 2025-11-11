import numpy as np
import pandas as pd
from tqdm import tqdm

class SwissMeteoFeature:
    def __init__(self, meteo_swiss_name, name):
        self.meteo_swiss_name = meteo_swiss_name
        self.name = name

    def open(self, meteo_dir):
        return pd.read_csv(f'{meteo_dir}/daily_{self.meteo_swiss_name}.csv', index_col=0)

class SwissMeteoFeatures:
    def __init__(self, metadata_df, wt_stations_metadata_df, meteo_dir):
        self.features = [
            SoilTemperatureAt10cmDepthDailyMean(),
            DiffuseRadiationDailyAverage(),
            EvapotranspirationDailyTotal(),
            CloudCoverDailyMean(),
            GlobalRadiationDailyMean(),
            SereneDayMeanCloucCoverBelow20Perc(),
            CoolingDegreeDay(),
            LongWaveIncomingRadiationDailyMean(),
            AtmosphericPressureAtBarometricAltitudeDailyMean(),
            PressureReducedToSeaLevelDailyMean(),
            PressureReducedToSeaLevelAccordingToStandardAtmosphereDailyMean(),
            FreshSnow2DayTotal(),
            PrecipitationDailyTotal(),
            ReferenceEvaporationFromFAODailyTotal(),
            RelativeAirHumidity2mAboveGroundDailyMean(),
            DayWithMeanCloudCoverBetween20percAnd60perc(),
            DayWithMeanCloudCoverBetween61percAnd80perc(),
            DayWithFog(),
            DayWithRain(),
            DayWithRainAndSnow(),
            DayWithSnowConverage(),
            DayWithSnowfall(),
            DarkDayMeanCloudCoverOver80perc(),
            WindSpeedScalarDailyMean(),
            WindSpeedMaxHourlyMeanOfTheDay(),
            ShortwaveReflectedRadiationDailyMean(),
            LongwaveOutgoingRadiationDailyMean(),
        ]

        self.meteo_dir = meteo_dir
        self.meteo_stations_weights_df = self.meteo_stations_weights_at_wt_stations(wt_stations_metadata_df, metadata_df) 

    def load(self):
        return {
            feature.name: feature.open(self.meteo_dir) for feature in self.features
        }

    def meteo_stations_weights_at_wt_stations(self, wt_stations_metadata_df, metadata_df):
        return pd.DataFrame.from_dict({
            row: {
                meteo_row: self._interpolation_weight(meteo_station['Coord X'], meteo_station['Coord Y'], station.lv95_Longitude, station.lv95_Latitude)
                for meteo_row, meteo_station in metadata_df.iterrows()
            }
            for row, station in wt_stations_metadata_df.iterrows()
        }).T

    def _interpolation_weight(self, lon, lat, target_lon, target_lat, beta=2):
        euclidean_dist = np.linalg.norm(
            np.array([lon, lat]) - np.array([target_lon, target_lat])
        )

        # In IDW, weights are 1 / distance
        # weights = 1.0/(dist+1e-12)**power
        return euclidean_dist ** (-beta)

    def get_features_at_stations(self):
        return pd.concat([
            self.get_features_at_station(station)
            for station in tqdm(self.meteo_stations_weights_df.index)
        ])

    def get_features_at_station(self, station):        
        def interpolate(row):
            row = row.dropna()

            if len(row) > 0:
                return np.average(row, weights=self.meteo_stations_weights_df.loc[station].loc[row.index])
            else:
                return float('nan')

        return pd.concat([
            pd.DataFrame({
                'feature': feature_name,
                'station': station,
                'interpolated': feature_df.apply(lambda row: interpolate(row), axis=1),
            })
            for feature_name, feature_df in self.load().items()
        ])

class SoilTemperatureAt10cmDepthDailyMean(SwissMeteoFeature):
    def __init__(self):
        super().__init__(meteo_swiss_name="tso010d0", name="Soil_temperature_at_10_cm_depth_daily_mean")

class DiffuseRadiationDailyAverage(SwissMeteoFeature):
    def __init__(self):
        super().__init__(meteo_swiss_name="ods000d0", name="Diffuse_radiation_daily_average")

class EvapotranspirationDailyTotal(SwissMeteoFeature):
    def __init__(self):
        super().__init__(meteo_swiss_name="ets150d0", name="Evapotranspiration_daily_total")

class CloudCoverDailyMean(SwissMeteoFeature):
    def __init__(self):
        super().__init__(meteo_swiss_name="nto000d0", name="Cloud_cover_daily_mean")

class GlobalRadiationDailyMean(SwissMeteoFeature):
    def __init__(self):
        super().__init__(meteo_swiss_name="gre000d0", name="Global_radiation_daily_mean")

class SereneDayMeanCloucCoverBelow20Perc(SwissMeteoFeature):
    def __init__(self):
        super().__init__(meteo_swiss_name="nto002d0", name="Serene_day_Mean_cloud_cover_below_20%")

class CoolingDegreeDay(SwissMeteoFeature):
    def __init__(self):
        super().__init__(meteo_swiss_name="xcd000d0", name="Cooling_Degree_Day_(CDD)")

class LongWaveIncomingRadiationDailyMean(SwissMeteoFeature):
    def __init__(self):
        super().__init__(meteo_swiss_name="oli000d0", name="Longwave_incoming_radiation_daily_mean")

class AtmosphericPressureAtBarometricAltitudeDailyMean(SwissMeteoFeature):
    def __init__(self):
        super().__init__(meteo_swiss_name="prestad0", name="Atmospheric_pressure_at_barometric_altitude_(QFE)_daily_mean")

class PressureReducedToSeaLevelDailyMean(SwissMeteoFeature):
    def __init__(self):
        super().__init__(meteo_swiss_name="pp0qffd0", name="Pressure_reduced_to_sea_level_(QFF)_daily_mean")

class PressureReducedToSeaLevelAccordingToStandardAtmosphereDailyMean(SwissMeteoFeature):
    def __init__(self):
        super().__init__(meteo_swiss_name="pp0qnhd0", name="Pressure_reduced_to_sea_level_according_to_standard_atmosphere_(QNH)_daily_mean")

class FreshSnow2DayTotal(SwissMeteoFeature):
    def __init__(self):
        super().__init__(meteo_swiss_name="hns002d0", name="Fresh_snow_2-day-total_-_48_h")

class PrecipitationDailyTotal(SwissMeteoFeature):
    def __init__(self):
        super().__init__(meteo_swiss_name="rka150d0", name="Precipitation_daily_total_0_UTC_-_0_UTC")

class ReferenceEvaporationFromFAODailyTotal(SwissMeteoFeature):
    def __init__(self):
        super().__init__(meteo_swiss_name="erefaod0", name="Reference_evaporation_from_FAO_daily_total")

class RelativeAirHumidity2mAboveGroundDailyMean(SwissMeteoFeature):
    def __init__(self):
        super().__init__(meteo_swiss_name="ure200d0", name="Relative_air_humidity_2_m_above_ground_daily_mean")

class DayWithMeanCloudCoverBetween20percAnd60perc(SwissMeteoFeature):
    def __init__(self):
        super().__init__(meteo_swiss_name="nto026d0", name="Day_with_mean_cloud_cover_between_20%_and_60%")

class DayWithMeanCloudCoverBetween61percAnd80perc(SwissMeteoFeature):
    def __init__(self):
        super().__init__(meteo_swiss_name="nto068d0", name="Day_with_mean_cloud_cover_between_61%_and_80%")

class DayWithFog(SwissMeteoFeature):
    def __init__(self):
        super().__init__(meteo_swiss_name="w5p002d0", name="Day_with_fog")

class DayWithRain(SwissMeteoFeature):
    def __init__(self):
        super().__init__(meteo_swiss_name="w1p012d0", name="Day_with_rain")

class DayWithRainAndSnow(SwissMeteoFeature):
    def __init__(self):
        super().__init__(meteo_swiss_name="w2p001d0", name="Day_with_rain_and_snow")
 
class DayWithSnowConverage(SwissMeteoFeature):
    def __init__(self):
        super().__init__(meteo_swiss_name="est000d0", name="Day_with_snow_coverage")

class DayWithSnowfall(SwissMeteoFeature):
    def __init__(self):
        super().__init__(meteo_swiss_name="w2p002d0", name="Day_with_snowfall")

class DarkDayMeanCloudCoverOver80perc(SwissMeteoFeature):
    def __init__(self):
        super().__init__(meteo_swiss_name="nto008d0", name="Dark_day_Mean_cloud_cover_over_80%")

class WindSpeedScalarDailyMean(SwissMeteoFeature):
    def __init__(self):
        super().__init__(meteo_swiss_name="fkl010d0", name="Wind_speed_scalar_daily_mean")

class WindSpeedMaxHourlyMeanOfTheDay(SwissMeteoFeature):
    def __init__(self):
        super().__init__(meteo_swiss_name="fhh010dx", name="Wind_speed_maximum_hourly_mean_of_the_day")

class ShortwaveReflectedRadiationDailyMean(SwissMeteoFeature):
    def __init__(self):
        super().__init__(meteo_swiss_name="osr000d0", name="Shortwave_reflected_radiation_daily_mean")

class LongwaveOutgoingRadiationDailyMean(SwissMeteoFeature):
    def __init__(self):
        super().__init__(meteo_swiss_name="olo000d0", name="Longwave_outgoing_radiation_daily_mean")