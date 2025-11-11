import pandas as pd


class WaterTemperatureLoader:
    def __init__(self, file_path, time_interval, sampling_quantile):
        self.file_path = file_path
        self.time_interval = time_interval
        self.sampling_quantile = sampling_quantile

    def resample(self, data):
        return data.resample(self.time_interval).quantile(self.sampling_quantile)

    def load(self):
        pass


class BoironWaterTemperatureLoader(WaterTemperatureLoader):
    def __init__(self, file_path, time_interval, sampling_quantile):
        super().__init__(file_path, time_interval, sampling_quantile)

    def load(self):
        data = pd.read_csv(self.file_path)
        data["Date et heure"] = pd.to_datetime(data["Date et heure"], dayfirst=True)
        data = data.set_index("Date et heure")
        data = self.resample(data)
        return data


class BroyeWaterTemperatureLoader(WaterTemperatureLoader):
    def __init__(self, file_path, time_interval, sampling_quantile):
        super().__init__(file_path, time_interval, sampling_quantile)

    def load(self):
        data = pd.read_csv(self.file_path)
        data = data.rename({"Unnamed: 0": "Date et heure"}, axis=1)
        data["Date et heure"] = pd.to_datetime(data["Date et heure"], dayfirst=False)
        data = data.set_index("Date et heure")
        data = self.resample(data)
        return data


class VenogeWaterTemperatureLoader(WaterTemperatureLoader):
    def __init__(self, file_path, time_interval, sampling_quantile):
        super().__init__(file_path, time_interval, sampling_quantile)

    def load(self):
        data = pd.read_csv(self.file_path)
        data = data.rename({"Unnamed: 0": "Date et heure"}, axis=1)
        data["Date et heure"] = pd.to_datetime(data["Date et heure"], dayfirst=False)
        data = data.set_index("Date et heure")
        data = self.resample(data)
        return data


class VaudWaterTemperatureLoader(WaterTemperatureLoader):
    def __init__(self, file_path, time_interval, sampling_quantile, other_df_columns):
        super().__init__(file_path, time_interval, sampling_quantile)
        self.other_df_columns = other_df_columns

    def load(self):
        data = pd.read_excel(self.file_path, sheet_name="15min")
        data["Date et heure"] = pd.to_datetime(data["Date et heure"], dayfirst=False)
        data = data.set_index("Date et heure").drop(["Date", "heure"], axis=1)
        data = self.resample(data)
        data = self.filter_duplicates(data)
        return data

    def filter_duplicates(self, data):
        return data[data.columns[~data.columns.isin(self.other_df_columns)]]
