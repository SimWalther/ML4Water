# Define rules for each group: necessary and unwanted keywords


# group_definitions = {
#     "Group_VHM_upstream": {
#         "type": "softmax",
#         "necessary": ["VHM_upstream"],
#         "unwanted": []
#     },

#     "Group_VHM_halfcircle": {
#         "type": "softmax",
#         "necessary": ["VHM_halfcircle"],
#         "unwanted": []
#     },

#     "Group_air_temperature": {
#         "type": "softmax",
#         "necessary": ["temperature", "Ta", "radiation"],
#         "unwanted": []
#     },


#     "Group_pressure": {
#         "type": "softmax",
#         "necessary": ["Pressure_reduced_to_sea_level_(QFF)_daily_mean",
#                       "Pressure_reduced_to_sea_level_according_to_standard_atmosphere_(QNH)_daily_mean",
#                       "Atmospheric_pressure_at_barometric_altitude_(QFE)_daily_mean"],
#         "unwanted": []
#     },

#     "Group_evaporation": {
#         "type": "softmax",
#         "necessary": ["Reference_evaporation_from_FAO_daily_total", "Evapotranspiration_daily_total"],
#         "unwanted": []
#     },

#     "Group_wind": {
#         "type": "softmax",
#         "necessary": ["Wind_speed_maximum_hourly_mean_of_the_day", "Wind_speed_scalar_daily_mean"],
#         "unwanted": []
#     },

#     "Group_cloud": {
#         "type": "softmax",
#         "necessary": [
#         "Cloud_cover_daily_mean",
#         "Serene_day_Mean_cloud_cover_below_20%",
#         "Dark_day_Mean_cloud_cover_over_80%",
#         "Day_with_mean_cloud_cover_between_20%_and_60%",
#         "Day_with_mean_cloud_cover_between_61%_and_80%"],
#         "unwanted": []
#     },

#     "Group_rain_fog_snow": {
#         "type": "softmax",
#         "necessary": ["Fresh_snow_2-day-total_-_48_h",
#                         "Precipitation_daily_total_0_UTC_-_0_UTC",
#                         "Day_with_rain",
#                         "Day_with_rain_and_snow",
#                         "Day_with_snowfall",
#                         "Day_with_fog",
#                         "Day_with_snow_coverage"],
#         "unwanted": []
#     },
# }


group_definitions = {
    "Group_VHM_upstream_distance100": {
        "type": "gaussian",
        "necessary": ["VHM_upstream"],
        "unwanted": ["distance250", "distance500", "distance1000"],
    },
    "Group_VHM_upstream_distance250": {
        "type": "gaussian",
        "necessary": ["VHM_upstream", "distance250"],
        "unwanted": ["distance100", "distance500", "distance1000"],
    },
    "Group_VHM_upstream_distance500": {
        "type": "gaussian",
        "necessary": ["VHM_upstream", "distance500"],
        "unwanted": ["distance100", "distance250", "distance1000"],
    },
    "Group_VHM_upstream_distance1000": {
        "type": "gaussian",
        "necessary": ["VHM_upstream", "distance1000"],
        "unwanted": ["distance100", "distance250", "distance500"],
    },
    "Group_VHM_halfcircle": {
        "type": "attention",
        "necessary": ["VHM_halfcircle"],
        "unwanted": [],
    },
    "Group_air_temperature": {
        "type": "attention",
        "necessary": [
            "air_max_temp",
            "temperature",
            "Ta",
            "radiation",
            "Cooling_Degree_Day_(CDD)",
        ],
        "unwanted": [],
    },
    "Group_pressure": {
        "type": "attention",
        "necessary": [
            "Pressure_reduced_to_sea_level_(QFF)_daily_mean",
            "Pressure_reduced_to_sea_level_according_to_standard_atmosphere_(QNH)_daily_mean",
            "Atmospheric_pressure_at_barometric_altitude_(QFE)_daily_mean",
        ],
        "unwanted": [],
    },
    "Group_evaporation": {
        "type": "attention",
        "necessary": [
            "Reference_evaporation_from_FAO_daily_total",
            "Evapotranspiration_daily_total",
        ],
        "unwanted": [],
    },
    "Group_wind": {
        "type": "attention",
        "necessary": [
            "Wind_speed_maximum_hourly_mean_of_the_day",
            "Wind_speed_scalar_daily_mean",
        ],
        "unwanted": [],
    },
    "Group_cloud": {
        "type": "attention",
        "necessary": [
            "Cloud_cover_daily_mean",
            "Serene_day_Mean_cloud_cover_below_20%",
            "Dark_day_Mean_cloud_cover_over_80%",
            "Day_with_mean_cloud_cover_between_20%_and_60%",
            "Day_with_mean_cloud_cover_between_61%_and_80%",
        ],
        "unwanted": [],
    },
    "Group_rain_fog_snow": {
        "type": "attention",
        "necessary": [
            "Fresh_snow_2-day-total_-_48_h",
            "Precipitation_daily_total_0_UTC_-_0_UTC",
            "Day_with_rain",
            "Day_with_rain_and_snow",
            "Day_with_snowfall",
            "Day_with_fog",
            "Day_with_snow_coverage",
        ],
        "unwanted": [],
    },
}


features_mapping_dict_total = {
    "tso010d0": "Soil_temperature_at_10_cm_depth_daily_mean",
    "ods000d0": "Diffuse_radiation_daily_average",
    "ets150d0": "Evapotranspiration_daily_total",
    "nto000d0": "Cloud_cover_daily_mean",
    "gre000d0": "Global_radiation_daily_mean",
    "nto002d0": "Serene_day_Mean_cloud_cover_below_20%",
    "xcd000d0": "Cooling_Degree_Day_(CDD)",
    "oli000d0": "Longwave_incoming_radiation_daily_mean",
    "prestad0": "Atmospheric_pressure_at_barometric_altitude_(QFE)_daily_mean",
    "pp0qffd0": "Pressure_reduced_to_sea_level_(QFF)_daily_mean",
    "pp0qnhd0": "Pressure_reduced_to_sea_level_according_to_standard_atmosphere_(QNH)_daily_mean",
    "hns002d0": "Fresh_snow_2-day-total_-_48_h",
    "rka150d0": "Precipitation_daily_total_0_UTC_-_0_UTC",
    "erefaod0": "Reference_evaporation_from_FAO_daily_total",
    "ure200d0": "Relative_air_humidity_2_m_above_ground_daily_mean",
    "nto026d0": "Day_with_mean_cloud_cover_between_20%_and_60%",
    "nto068d0": "Day_with_mean_cloud_cover_between_61%_and_80%",
    "w5p002d0": "Day_with_fog",
    "w1p012d0": "Day_with_rain",
    "w2p001d0": "Day_with_rain_and_snow",
    "est000d0": "Day_with_snow_coverage",
    "w2p002d0": "Day_with_snowfall",
    "nto008d0": "Dark_day_Mean_cloud_cover_over_80%",
    "fkl010d0": "Wind_speed_scalar_daily_mean",
    "fhh010dx": "Wind_speed_maximum_hourly_mean_of_the_day",
    "osr000d0": "Shortwave_reflected_radiation_daily_mean",
    "olo000d0": "Longwave_outgoing_radiation_daily_mean",
    "NVDI": "NDVI",
}


mapping_dict_station_to_river = {
    "BOI011": "Boiron",
    "BOI013": "Boiron",
    "BOI020": "Boiron",
    "RTV028": "Boiron",
    "BOI018": "Boiron",
    "BOI019": "Boiron",
    "BOI015": "Boiron",
    "D002": "Dullive",
    "D001": "Dullive",
    "RTV018": "Dullive",
    "RTV063": "Hongrin",
    "RTV062": "Torneresse",
    "RTV003": "Aubonne",
    "MRP001": "Aubonne",
    "RTV072": "Broye",
    "RTV015": "Broye",
    "RTV022": "Broye",
    "RTV070": "Broye",
    "RTV046": "Broye",
    "RTV074": "Mentue",
    "RTV030": "Mentue",
    "RTV037": "Nozon",
    "RTV035": "Nozon",
    "RTV036": "Nozon",
    "RTV077": "Orbe",
    "RTV076": "Orbe",
    "RTV040": "Orbe",
    "RTV092": "Orbe",
    "RTV039": "Orbe",
    "RTV055": "Venoge",
    "RTV054": "Venoge",
    "RTV053": "Venoge",
    "RTV078": "Venoge",
    "MRP005": "Venoge",
    "RTV061": "Veyron",
    "RTV060": "Veyron",
    "RTV085": "Veyron",
    "RTV024": "Forestay",
    "RTV029": "Lutrive",
    "RTV041": "Paudèze",
    "RTV043": "Promenthouse",
    "RTV047": "Serine",
    "RTV050": "Talent",
    "RTV052": "Vaux",
    "RTV080": "Versoix",
    "RTV071": "Broye",
    "RTV014": "Broye",
    "RTV069": "Broye",
    "RTV068": "Broye",
    "RTV073": "Broye",
    "RTV067": "Broye",
    "RTV066": "Broye",
    "RTV065": "Broye",
    "RTV013": "Broye",
    "RTV064": "Bioleyres",
    "BOI021": "Boiron",
    "BOI012": "Boiron",
    "RTV008": "Boiron",
    "BOI009": "Boiron",
    "RTV045": "Boiron",
    "BOI006": "Boiron",
    "BOI005": "Boiron",
    "BOI001": "Boiron",
    "RTV021": "Boiron",
    "BOI017": "Boiron",
    "BOI008": "Boiron",
    "BOI016": "Boiron",
}
