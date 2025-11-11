import pandas as pd

class AirTemperatureLoader:
    def __init__(self, file_path, time_interval, sampling_quantile):
        self.file_path = file_path
        self.time_interval = time_interval
        self.sampling_quantile = sampling_quantile

    def resample(self, data):
        return data.resample(self.time_interval).quantile(self.sampling_quantile)

    def load(self):
        pass

class BoironAirTemperatureLoader(AirTemperatureLoader):
    def __init__(self, file_path, time_interval, sampling_quantile):
        super().__init__(file_path, time_interval, sampling_quantile)

    def load(self):
        data = pd.read_csv(self.file_path)
        data = data.rename({"Unnamed: 0": 'date'}, axis=1)
        data['date'] = pd.to_datetime(data['date'], dayfirst=False)
        data = data.set_index('date').resample('1D').max()
        return data

class BroyeAirTemperatureLoader(AirTemperatureLoader):
    def __init__(self, file_path, time_interval, sampling_quantile):
        super().__init__(file_path, time_interval, sampling_quantile)
    
    def load(self):
        data = pd.read_csv(self.file_path)
        data = data.rename({"Unnamed: 0": 'date'}, axis=1)
        data['date'] = pd.to_datetime(data['date'], dayfirst=False)
        data = data.set_index('date').resample('1D').max()
        return data

class VenogeAirTemperatureLoader(AirTemperatureLoader):
    def __init__(self, file_path, time_interval, sampling_quantile):
        super().__init__(file_path, time_interval, sampling_quantile)
    
    def load(self):    
        data = pd.read_csv(self.file_path)
        data = data.rename({"Unnamed: 0": 'date'}, axis=1)
        data['date'] = pd.to_datetime(data['date'], dayfirst=False)
        data = data.set_index('date').resample('1D').max()
        return data

class VaudAirTemperatureLoader(AirTemperatureLoader):
    def __init__(self, file_path, time_interval, sampling_quantile, other_df_columns):
        super().__init__(file_path, time_interval, sampling_quantile)
        self.other_df_columns = other_df_columns
    
    def load(self):
        data = pd.read_csv(self.file_path)
        data = data.rename({"Unnamed: 0": 'date'}, axis=1)
        data['date'] = pd.to_datetime(data['date'], dayfirst=False)
        data = data.set_index('date').resample('1D').max()
        data = self.filter_duplicates(data)
        return data

    def filter_duplicates(self, data):
        return data[data.columns[~data.columns.isin(self.other_df_columns)]]