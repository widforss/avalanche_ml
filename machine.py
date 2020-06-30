import datetime as dt
import os
import pickle
import re
import sys
import time
from collections import OrderedDict
from aggregatedata import ForecastDataset, LabeledData, REG_ENG, CsvMissingError

import numpy as np
import pandas
import requests
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential

sys.path.insert(0, "./varsomdata")
import setenvironment as se
from varsomdata import getforecastapi as gf
from varsomdata import getvarsompickles as gvp
from varsomdata import getmisc as gm


__author__ = 'arwi'

_ap = lambda x: f"avalanche_problem_{x}_"
_NONE = "none"
LABEL_GLOBAL = {
    "danger_level": {
        "train": True,
        "ext_attr": ["danger_level", "danger_level_name"],
        "values": {
            1: [1, "1 liten"],
            2: [2, "2 Moderat"],
            3: [3, "3 Betydelig"],
            4: [4, "4 Stor"],
            5: [5, "5 Meget stor"]
        }
    },
    "emergency_warning": {
        "train": True,
        "ext_attr": [["emergency_warning"]],
        "values": {
           "Ikke gitt": ["Ikke gitt"],
           "Naturlig utløste skred": ["Naturlig utløste skred"],
        }
    },
    "problem_amount": {
        "train": False,
        "ext_attr": [f"avalanche_problem_{n}_problem_id" for n in range(1, 4)],
        "values": {
            0: [0, 0, 0],
            1: [1, 0, 0],
            2: [1, 2, 0],
            3: [1, 2, 3]
        }
    }
}
for n in range(1, 4):
    LABEL_GLOBAL[f"problem_{n}"] = {
        "train": True,
        "ext_attr": [f"{_ap(n)}{attr}" for attr in [
            "problem_type_id", "problem_type_name", "type_id", "type_name", "ext_id", "ext_name"
        ]],
        "values": {
            _NONE: [0, "", 0, "", 0, ""],
            "new-loose": [3, "Nysnø (løssnøskred)", 20, "Løssnøskred", 10, "Tørre løssnøskred"],
            "wet-loose": [5, "Våt snø (løssnøskred)", 20, "Løssnøskred", 15, "Våte løssnøskred"],
            "new-slab": [7, "Nysnø (flakskred)", 10, "Flakskred", 20, "Tørre flakskred"],
            "drift-slab": [10, "Fokksnø (flakskred)", 10, "Flakskred", 20, "Tørre flakskred"],
            "pwl-slab": [30, "Vedvarende svakt lag (flakskred)", 10, "Flakskred", 20, "Tørre flakskred"],
            "wet-slab": [45, "Våt snø (flakskred)", 10, "Flakskred", 25, "Våte flakskred"],
            "glide": [50, "Glideskred", 10, "Flakskred", 25, "Våte flakskred"]
        }
    }

LABEL_PROBLEM = {
    "cause": {
        "train": False,
        "ext_attr": ["cause_id", "cause_name"],
        "values": {
            "new-snow": [10, "Nedføyket svakt lag med nysnø"],
            "hoar": [11, "Nedsnødd eller nedføyket overflaterim"],
            "facet": [13, "Nedsnødd eller nedføyket kantkornet snø"],
            "crust": [14, "Dårlig binding mellom glatt skare og overliggende snø"],
            "snowdrift": [15, "Dårlig binding mellom lag i fokksnøen"],
            "ground-facet": [16, "Kantkornet snø ved bakken"],
            "crust-above-facet": [18, "Kantkornet snø over skarelag"],
            "crust-below-facet": [19, "Kantkornet snø under skarelag"],
            "ground-water": [20, "Vann ved bakken/smelting fra bakken"],
            "loose": [22, "Opphopning av vann i/over lag i snødekket"],
            "rain-temp-sun": [24, "Ubunden snø"]
        }
    },
    "dsize": {
        "train": False,
        "ext_attr": ["destructive_size_ext_id", "destructive_size_ext_name"],
        "values": {
            1: [1, "1 - Små"],
            2: [2, "2 - Middels"],
            3: [3, "3 - Store"],
            4: [4, "4 - Svært store"],
            5: [5, "5 - Ekstremt store"]
        }
    },
    "prob": {
        "train": False,
        "ext_attr": ["probability_id", "probability_name"],
        "values": {
            2: [2, "Lite sannsynlig"],
            3: [3, "Mulig"],
            5: [5, "Sannsynlig"],
        }
    },
    "trig": {
        "train": False,
        "ext_attr": ["trigger_simple_id", "trigger_simple_name"],
        "values": {
            10: [10, "Stor tilleggsbelastning"],
            21: [21, "Liten tilleggsbelastning"],
            22: [22, "Naturlig utløst"]
        }
    },
    "dist": {
        "train": False,
        "ext_attr": ["distribution_id", "distribution_name"],
        "values": {
            1: [1, "Få bratte heng"],
            2: [2, "Noen bratte heng"],
            3: [3, "Mange bratte heng"],
            4: [4, "De fleste bratte heng"]
        }
    },
    "lev_fill": {
        "train": False,
        "ext_attr": ["avalanche_problem_1_exposed_height_fill"],
        "values": {
            1: [1],
            2: [2],
            3: [3],
            4: [4],
        }
    }
}


class BulletinMachine:
    def __init__(self, ml_class_creator):
        self.ml_class_creator = ml_class_creator
        self.machines_global = {}
        self.machines_problem = {}
        self.labels = {}
        self.X = None
        self.y = None
        self.is_timeseries = False
        self.fitted = False

    def fit(self, labeled_data, is_timeseries, epochs, verbose=0):
        if self.fitted:
            raise AlreadyFittedError()
        self.fitted = True
        self.is_timeseries = is_timeseries
        labeled_data.normalize()
        self.y = labeled_data.label
        self.X = labeled_data.to_timeseries()[0] if is_timeseries else labeled_data.data.values
        for column in labeled_data.label.columns:
            if column[0] in LABEL_GLOBAL and LABEL_GLOBAL[column[0]]["train"]:
                self.labels[column[0]] = pandas.DataFrame(
                    pandas.get_dummies(labeled_data.label[column]),
                    columns=list(LABEL_GLOBAL[column[0]]["values"].keys())
                ).fillna(0)
        for label, dummies in self.labels.items():
            self.machines_global[label] = self.ml_class_creator(self.X.shape[1:], len(dummies.columns))

        for attr, machine in self.machines_global.items():
            machine.fit(self.X, self.labels[attr], epochs=epochs, verbose=verbose)

    def predict(self, unlabeled_data):
        def _predict_class(attr, machine_scope, label_scope):
            machine = machine_scope[attr]
            idx = np.argmax(machine.predict(X), axis=1)
            y[(attr, "CLASS")] = np.array(list(LABEL_GLOBAL[attr]["values"].keys()))[idx]

        if not self.fitted:
            raise NotFittedError()

        unlabeled_data.normalize()
        X = unlabeled_data.to_timeseries()[0] if self.is_timeseries else unlabeled_data.data.values
        y = pandas.DataFrame(index=unlabeled_data.data.index, columns=self.y.columns)
        y = y.fillna(0).astype(self.y.dtypes.to_dict())
        for attr in ["danger_level", "emergency_warning"]:
            _predict_class(attr, self.machines_global, LABEL_GLOBAL)

        last_argmax = None
        for n in range(1, 4):
            argmax = np.argmax(self.machines_global[f"problem_{n}"].predict(X), axis=1)
            # Set to None if problem_n-1 was None.
            argmax = argmax if last_argmax is None else argmax * last_argmax.astype(np.bool)
            last_argmax = argmax
            y[("problem_amount", "CLASS")] = y[("problem_amount", "CLASS")] + argmax.astype(np.bool)
            y[(f"problem_{n}", "CLASS")] = np.array(list(LABEL_GLOBAL[f"problem_{n}"]["values"].keys()))[argmax]
        return y


class Error(Exception):
    pass


class AlreadyFittedError(Error):
    pass


class NotFittedError(Error):
    pass


if __name__ == '__main__':
    def machine_creator(indata, outdata):
        model = Sequential()
        model.add(LSTM(200, activation='tanh', input_shape=indata))
        model.add(Dense(units=outdata, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        return model

    days = 7
    regobs_types = list(REG_ENG.keys())
    print("Reading csv")
    labeled_data = None
    try:
        labeled_data = LabeledData.from_csv(days=days, regobs_types=regobs_types)
    except CsvMissingError:
        forecast_dataset = ForecastDataset(regobs_types=regobs_types)
        labeled_data = forecast_dataset.label(days=days)
        labeled_data.to_csv()

    print("Running classifier")
    kf = KFold(n_splits=5, shuffle=True)
    for split_idx, (train_index, test_index) in enumerate(kf.split(labeled_data.data)):
        training_data = labeled_data.copy()
        training_data.data = labeled_data.data.iloc[train_index]
        training_data.label = labeled_data.label.iloc[train_index]
        testing_data = labeled_data.copy()
        testing_data.data = labeled_data.data.iloc[test_index]
        testing_data.label = labeled_data.label.iloc[test_index]

        bm = BulletinMachine(machine_creator)
        bm.fit(training_data, True, 80)
        predictions = bm.predict(testing_data)
        predictions.to_csv("pred.csv", sep=';')

        for column in ["danger_level", "emergency_warning", "problem_amount", "problem_1", "problem_2", "problem_3"]:
            print(column)
            weighted_f1 = f1_score(testing_data.label[column], predictions[column], average='weighted')
            macro_f1 = f1_score(testing_data.label[column], predictions[column], average='macro')
            print(f"Weighted F1 score for {column} in split {split_idx}: {weighted_f1}")
            print(f"Macro F1 score for {column} in split {split_idx}: {macro_f1}")
