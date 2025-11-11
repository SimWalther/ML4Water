def trig_day_from_dayofyear(dayofyear):
    day_sin = (np.sin(2 * np.pi * dayofyear / 366) + 1) / 2
    day_cos = (np.cos(2 * np.pi * dayofyear / 366) + 1) / 2

    return day_cos, day_sin