# -*- coding: utf-8 -*-
"""Structures data in ML-friendly ways."""

import sys
import datetime as dt
import csv
import numpy as np
import pandas
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, "./varsomdata")
import setenvironment as se
from varsomdata import getforecastapi as gf
from varsomdata import getvarsompickles as gvp

__author__ = 'arwi'

WIND_DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

WIND_SPEEDS = {
    'Stille/svak vind': 0.,
    'Bris': 5.5,
    'Frisk bris': 9.,
    'Liten kuling': 12.,
    'Stiv kuling': 15.5,
    'Sterk kuling': 18.5,
    'Liten storm': 23.,
    'Storm': 30.
}

PROBLEMS = {
    3: 'new_loose',
    5: 'wet_loose',
    7: 'new_slab',
    10: 'drift_slab',
    30: 'pwl_slab',
    37: 'pwl_slab',
    45: 'wet_slab',
    50: 'glide'
}

CAUSES = {
    10: 'new_wl',
    11: 'rime',
    13: 'facet',
    14: 'ice',
    15: 'drift',
    16: 'gnd_facet',
    18: 'ice_a_facet ',
    19: 'ice_b_facet',
    20: 'gnd_water',
    22: 'water',
    24: 'loose'
}

TRIGGERS = {
    10: 0,
    21: 1,
    22: 2
}


class ForecastDataset:

    def __init__(self):
        """
        Object contains aggregated data used to generate labeled datasets.
        """
        seasons = ['2016-17', '2017-18', '2018-19', '2019-20']
        aw = []
        for season in seasons:
            aw += gvp.get_all_forecasts(year=season)

        tree = {}
        flat = []
        for forecast in aw:
            if forecast.region_id not in tree:
                tree[forecast.region_id] = {}

            row = {
                # Metadata
                'region_id': forecast.region_id,
                'region_name': forecast.region_name,
                'region_type': forecast.region_type_name,
                'date': forecast.date_valid,

                'danger_level': forecast.danger_level,
                'emergency_warning': float(forecast.emergency_warning == 'Ikke gitt')
            }

            # Weather data
            weather = {
                'precip_most_exposed': forecast.mountain_weather.precip_most_exposed,
                'precip': forecast.mountain_weather.precip_region,
                'wind_speed': WIND_SPEEDS.get(forecast.mountain_weather.wind_speed, 0),
                'wind_change_speed': WIND_SPEEDS.get(forecast.mountain_weather.change_wind_speed, 0),
                'temp_min': forecast.mountain_weather.temperature_min,
                'temp_max': forecast.mountain_weather.temperature_max,
                'temp_lev': forecast.mountain_weather.temperature_elevation,
                'temp_freeze_lev': forecast.mountain_weather.freezing_level,
            }

            # We use multiple loops to get associated values near each other in e.g. .csv-files.
            for wind_dir in WIND_DIRECTIONS:
                weather[f"wind_dir_{wind_dir}"] = float(forecast.mountain_weather.wind_direction == wind_dir)
            for wind_dir in WIND_DIRECTIONS:
                weather[f"wind_chg_dir_{wind_dir}"] = float(forecast.mountain_weather.change_wind_direction == wind_dir)
            hours = [0, 6, 12, 18]
            for h in hours:
                weather[f"wind_chg_start_{h}"] = float(forecast.mountain_weather.change_hour_of_day_start == h)
            for h in hours:
                weather[f"temp_fl_start_{h}"] = float(forecast.mountain_weather.change_hour_of_day_start == h)
            row['weather'] = weather

            # Problem data
            prb = {}
            problem_types = [PROBLEMS.get(p.avalanche_problem_type_id, None) for p in forecast.avalanche_problems]
            problems = {}
            prb['problem_amount'] = len(forecast.avalanche_problems)
            for problem in PROBLEMS.values():
                exists = problem in problem_types
                index = problem_types.index(problem) if exists else None
                problems[problem] = forecast.avalanche_problems[index] if exists else gf.AvalancheWarningProblem()
                prb[f"problem_{problem}"] = -(problems[problem].avalanche_problem_id - 4) if exists else 0
            for problem in PROBLEMS.values():
                p_data = problems[problem]
                for cause in CAUSES.values():
                    forecast_cause = CAUSES.get(p_data.aval_cause_id, None)
                    prb[f"problem_{problem}_cause_{cause}"] = float(forecast_cause == cause)
                prb[f"problem_{problem}_dsize"] = p_data.destructive_size_ext_id
                prb[f"problem_{problem}_prob"] = p_data.aval_probability_id
                prb[f"problem_{problem}_trig"] = TRIGGERS.get(p_data.aval_trigger_simple_id, 0)
                prb[f"problem_{problem}_dist"] = p_data.aval_distribution_id
                prb[f"problem_{problem}_lev_max"] = p_data.exposed_height_1
                prb[f"problem_{problem}_lev_min"] = p_data.exposed_height_2
                for n in range(0, 8):
                    aspect_attr_name = f"problem_{problem}_aspect_{WIND_DIRECTIONS[n]}"
                    prb[aspect_attr_name] = float(p_data.valid_expositions[n])

                # Check for consistency
                if prb[f"problem_{problem}_lev_min"] > prb[f"problem_{problem}_lev_max"]:
                    continue

            row['problems'] = prb

            # Check for consistency
            if weather['temp_min'] > weather['temp_max']:
                continue

            tree[forecast.region_id][forecast.date_valid] = row
            flat.append(row)

        self.tree = tree
        self.flat = flat
        self.prepared_data = {}

    def label(self, column_format, days, start_date=dt.date(2017, 11, 29), b_regions=False):
        table = []
        labels = []

        for row in self.flat:
            prev = []
            if row['date'] < start_date or (not b_regions and row['region_type'] == 'B'):
                continue
            try:
                for n in range(0, days + 2):
                    prev.append(self.tree[row['region_id']][row['date'] - dt.timedelta(days=n)])
            except KeyError:
                continue

            label = {
                'danger_level': row['danger_level']
            }

            data = {}
            danger_list = ['danger_level', 'emergency_warning']
            if column_format:
                # 0-indexed to include current day ("prev_0")
                for n in range(0, days + 1):
                    for key in row['weather'].keys():
                        data[f"prev_{n}_{key}"] = prev[n]['weather'][key]
                for n in range(1, days + 2):
                    for key in danger_list:
                        data[f"prev_{n}_{key}"] = prev[n][key]
                    for key in row['problems'].keys():
                        data[f"prev_{n}_{key}"] = prev[n]['problems'][key]
            else:
                for key in list(row['weather'].keys()) + danger_list + list(row['problems'].keys()):
                    data[key] = np.array([]).astype(np.float64)
                for n in range(0, days + 1):
                    for key in row['weather'].keys():
                        data[key] = np.append(data[key], prev[n]['weather'][key])
                # Shift forecast forward one day to align attributes.
                for n in range(1, days + 2):
                    for key in danger_list:
                        data[key] = np.append(data[key], prev[n][key]) if n > 0 else np.append(data[key], 0)
                    for key in row['problems'].keys():
                        data[key] = np.append(data[key], prev[n]['problems'][key]) if n > 0 else np.append(data[key], 0)

            table.append(data)
            labels.append(label)

        df_label = pandas.DataFrame(labels)
        df = pandas.DataFrame(table).fillna(0)
        if column_format:
            df = df.astype(np.float64)
        return LabeledData(df, df_label, days, column_format)


class LabeledData:
    is_normalized = False

    def __init__(self, data, label, days, column_format):
        self.data = data
        self.label = label
        self.days = days
        self.column_format = column_format

    def to_csv(self):
        # Write training data
        col = "column" if self.column_format else "array"
        pathname_data = f"{se.local_storage}data_{self.days}_days_{col}.csv"
        pathname_label = f"{se.local_storage}label.csv"
        for (pathname, df) in [(pathname_data, self.data), (pathname_label, self.label)]:
            with open(pathname, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(df.columns)
                for index, row in df.iterrows():
                    writer.writerow(row)


if __name__ == '__main__':
    print("Fetching data")
    forecast_dataset = ForecastDataset()
    print("Labeling data")
    labeled_data = forecast_dataset.label(True, days=3)
    print("Writing .csv-file")
    labeled_data.to_csv()
    print("Transforming label")
    le = LabelEncoder()
    labels = labeled_data.label.values.ravel()
    le.fit(labels)
    labels = le.transform(labels)
    print("Running classifier")
    scaler = MinMaxScaler()
    scaler.fit(labeled_data.data)
    normalized_data = scaler.transform(labeled_data.data)
    X_train, X_test, y_train, y_test = train_test_split(normalized_data, labels, test_size=0.2, random_state=1)
    clf = SVC(random_state=0)
    clf.fit(X_train, y_train)
    predictions = le.inverse_transform(clf.predict(X_test))
    danger = le.inverse_transform(y_test)
    accuracy = (danger == predictions).sum() / danger.shape[0]
    print(confusion_matrix(danger, predictions, labels=le.classes_))
    print(f"Accuracy: {accuracy}")