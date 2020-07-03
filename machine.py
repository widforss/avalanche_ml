import sys
from aggregatedata import ForecastDataset, LabeledData, REG_ENG, PROBLEMS, DIRECTIONS, _NONE, CsvMissingError
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas

sys.path.insert(0, "./varsomdata")
import setenvironment as se
from varsomdata import getforecastapi as gf
from varsomdata import getvarsompickles as gvp
from varsomdata import getmisc as gm


__author__ = 'arwi'

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
        "ext_attr": [f"avalanche_problem_{n}_{attr}" for attr in [
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
        "train": True,
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
            "water-layers": [22, "Opphopning av vann i/over lag i snødekket"],
            "loose": [24, "Ubunden snø"]
        }
    },
    "dsize": {
        "train": True,
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
        "train": True,
        "ext_attr": ["probability_id", "probability_name"],
        "values": {
            2: [2, "Lite sannsynlig"],
            3: [3, "Mulig"],
            5: [5, "Sannsynlig"],
        }
    },
    "trig": {
        "train": True,
        "ext_attr": ["trigger_simple_id", "trigger_simple_name"],
        "values": {
            10: [10, "Stor tilleggsbelastning"],
            21: [21, "Liten tilleggsbelastning"],
            22: [22, "Naturlig utløst"]
        }
    },
    "dist": {
        "train": True,
        "ext_attr": ["distribution_id", "distribution_name"],
        "values": {
            1: [1, "Få bratte heng"],
            2: [2, "Noen bratte heng"],
            3: [3, "Mange bratte heng"],
            4: [4, "De fleste bratte heng"]
        }
    },
    "lev_fill": {
        "train": True,
        "ext_attr": ["exposed_height_fill"],
        "values": {
            1: [1],
            2: [2],
            3: [3],
            4: [4],
        }
    }
}

LABEL_PROBLEM_MULTI = {
    "aspect": {
        "train": True,
        "ext_attr": ["valid_expositions"],
        "classes": DIRECTIONS
    }
}

LABEL_PROBLEM_REAL = {
    #"lev_max": {
    #    "train": True,
    #    "ext_attr": "exposed_height_1",
    #    "offset": 0,
    #    "scale": 1800
    #},
    #"lev_min": {
    #    "train": True,
    #    "ext_attr": "exposed_height_2",
    #    "offset": 0,
    #    "scale": 1600
    #}
}


class BulletinMachine:
    def __init__(self, ml_class_creator, ml_multiclass_creator, ml_real_creator):
        """Facilitates training and prediction of avalanche warnings.

        :param ml_class_creator: fn(in_size: Int, out_size: Int) -> classifier, where model supports model.fit(X, y)
                                 and model.predict(X) that returns softmax array or 1-hot.
        :param ml_multiclass_creator: fn(in_size: Int, out_size: Int) -> classifier, where model supports
                                 model.fit(X, y) and model.predict(X) that returns k-hot.
        :param ml_real_creator: fn(in_size: Int, out_size: Int) -> regressor, where model supports model.fit(X, y)
                                 and model.predict(X) that returns k-hot.
        """
        self.ml_class_creator = ml_class_creator
        self.ml_multiclass_creator = ml_multiclass_creator
        self.ml_real_creator = ml_real_creator
        self.machines = {}
        self.labels_class = {}
        self.labels_multi = {}
        self.labels_real = {}
        self.X = None
        self.y = None
        self.is_timeseries = False
        self.fitted = False
        self.columns = []

    def fit(self, labeled_data, epochs, verbose=0):
        """

        :param labeled_data: LabeledData: Dataset that the models should be fit after.
        :param epochs: Int. Number of epochs to train. Ignored if the supplied model doesn't
                       support the parameter.
        :param verbose: Int. Verbosity of the models. Ignored if not supported of the supplied
                        models.
        """
        if self.fitted:
            raise AlreadyFittedError()
        self.fitted = True
        labeled_data = labeled_data.normalize()
        self.y = labeled_data.label.replace(np.nan, _NONE, regex=True)
        self.columns = labeled_data.data.columns
        for column in [prop[0] for prop in LABEL_GLOBAL.items() if prop[1]["train"]]:
            categories = list(LABEL_GLOBAL[column]["values"].keys())
            self.labels_class[column] = pandas.DataFrame(
                pandas.get_dummies(self.y[(column, "CLASS")], columns=categories),
                columns=categories
            ).fillna(0)
        for problem in PROBLEMS.values():
            for prop in [prop[0] for prop in LABEL_PROBLEM.items() if prop[1]["train"]]:
                column = f"problem_{problem}_{prop}"
                categories = list(LABEL_PROBLEM[prop]["values"].keys())
                self.labels_class[column] = pandas.DataFrame(
                    pandas.get_dummies(self.y[(column, "CLASS")], columns=categories),
                    columns=categories
                ).fillna(0)
            for prop in [prop[0] for prop in LABEL_PROBLEM_MULTI.items() if prop[1]["train"]]:
                column = f"problem_{problem}_{prop}"
                labels = np.char.zfill(self.y[(column, "MULTICLASS")].values.astype("U"), 8)
                labels = np.char.split(np.char.replace(np.char.replace(labels, "0", "0,"), "1", "1,"), ",", 7)
                labels = np.char.startswith(np.stack(labels), "1").astype(np.float)
                self.labels_multi[column] = pandas.DataFrame(labels).fillna(0)
            for prop, d in [(prop[0], prop[1]) for prop in LABEL_PROBLEM_REAL.items() if prop[1]["train"]]:
                column = f"problem_{problem}_{prop}"
                self.labels_real[column] = (self.y[column] - d["offset"]) / d["scale"]

        labels = [(i, "single") for i in self.labels_class.items()]
        labels += [(i, "multi") for i in self.labels_multi.items()]
        labels += [((key, pandas.DataFrame(columns=[1])), "real") for key in self.labels_multi.keys()]
        for (label, dummies), type in labels:
            self.X = labeled_data.data.values
            machine_creator = {
                "single": self.ml_class_creator,
                "multi": self.ml_multiclass_creator,
                "real": self.ml_real_creator
            }[type]
            try:
                self.machines[label] = machine_creator(self.X.shape[1:], len(dummies.columns))
            except ValueError:
                self.X = labeled_data.to_timeseries()[0]
                self.is_timeseries = True
                self.machines[label] = machine_creator(self.X.shape[1:], len(dummies.columns))

        for (attr, _), type in labels:
            machine = self.machines[attr]
            rel_labels = {
                "single": self.labels_class,
                "multi": self.labels_multi,
                "real": self.labels_real
            }[type]
            idx = np.sum(rel_labels[attr], axis=1).astype(np.bool)
            print(f"Fitting: {attr}")
            try:
                machine.fit(self.X[np.ix_(idx)], rel_labels[attr].loc[idx], epochs=epochs, verbose=verbose)
            except TypeError:
                machine.fit(self.X[np.ix_(idx)], rel_labels[attr].loc[idx])

    def predict(self, data):
        """

        :param data: LabeledData. Dataset to predict. May have empty LabeledData.label.
        :return: LabeledData. A copy of data, with LabeledData.pred filled in.
        """
        def _predict_class(rows, attr, label_attr, machine_scope, label_scope):
            machine = machine_scope[attr]
            idx = np.argmax(machine.predict(X[np.ix_(rows)]), axis=1)
            y.loc[rows, (attr, "CLASS")] = np.array(list(label_scope[label_attr]["values"].keys()))[idx]

        if not self.fitted:
            raise NotFittedError()

        data = data.normalize()
        X = data.to_timeseries()[0] if self.is_timeseries else data.data.values
        y = pandas.DataFrame(index=data.data.index, columns=self.y.columns).fillna(0).astype(self.y.dtypes.to_dict())
        y.loc[:, y.dtypes == np.object] = _NONE
        for attr in ["danger_level", "emergency_warning"]:
            _predict_class([True] * y.shape[0], attr, attr, self.machines, LABEL_GLOBAL)

        softmax = np.stack([self.machines[f"problem_{n}"].predict(X) for n in [1, 2, 3]], axis=1)
        idxs = np.flip(softmax.argsort(axis=2), axis=2)
        is_prob = np.any(np.sum(softmax.astype(np.bool), axis=2) > 1)
        for _ in [0, 1]:
            if is_prob:
                # If second likeliest problem_n-1 is very likely, use that instead
                fst = np.expand_dims(np.arange(y.shape[0]), axis=1)
                sec = [[0, 1]] * y.shape[0]
                likely = softmax[fst, sec, idxs[fst, sec, 1]] > 0.75 * softmax[fst, sec, idxs[fst, sec, 0]]
                idxs[:, 1:, 0] = idxs[:, 1:, 0] * np.invert(likely) + idxs[:, :-1, 1] * likely
                # If equal to problem_n-1/2, set to second likeliest alternative.
                prev_eq = idxs[:, 1:, :1] == idxs[:, :-1, :1]
                idxs[:, 1:, :-1] = idxs[:, 1:, :-1] * np.invert(prev_eq) + idxs[:, 1:, 1:] * prev_eq
            else:
                # If equal to problem_n-1/2, set to _NONE.
                prev_eq = idxs[:, 1:, :1] == idxs[:, :-1, :1]
                idxs[:, 1:, :-1] = idxs[:, 1:, :-1] * np.invert(prev_eq)
            # Set to None if problem_n-1/2 was None.
            idxs[:, 1:] = idxs[:, 1:] * idxs[:, :-1, :1].astype(np.bool)
        idxs = idxs[:, :, 0]
        labels = np.array(list(LABEL_GLOBAL["problem_1"]["values"].keys()))[np.ravel(idxs)].reshape(idxs.shape)
        y[[(f"problem_{n}", "CLASS") for n in [1, 2, 3]]] = labels
        y[("problem_amount", "CLASS")] = np.sum(idxs.astype(np.bool), axis=1)
        print(np.sum(np.char.equal(y[("problem_3", "CLASS")].values.astype("U"), _NONE)))

        for problem in PROBLEMS.values():
            problem_cols = [(f"problem_{n}", "CLASS") for n in range(1, 4)]
            relevant_rows = np.any(np.equal(y[problem_cols].astype("U"), np.array([problem]).astype("U")), axis=1)
            if np.sum(relevant_rows):
                for attr in self.labels_class.keys():
                    machine_attr = f"problem_{problem}_{attr}"
                    _predict_class(relevant_rows, machine_attr, attr, self.machines, LABEL_PROBLEM)
                for attr in self.labels_multi.keys():
                    attr = f"problem_{problem}_{attr}"
                    classes = self.machines[attr].predict(X[np.ix_(relevant_rows)]).astype(np.int).astype("U")
                    y.loc[:, (attr, "MULTICLASS")] = "0" * classes.shape[1]
                    y.loc[relevant_rows, (attr, "MULTICLASS")] = [''.join(row)[:-1] for row in classes]
                for attr in self.labels_real.keys():
                    y.loc[:, (attr, "REAL")] = 0
                    pred = np.clip(self.machines[attr].predict(X[np.ix_(relevant_rows)]), 0, 1)
                    pred = pred * LABEL_PROBLEM_REAL[attr]["scale"] + LABEL_PROBLEM_REAL[attr]["offset"]
                    y.loc[relevant_rows, (attr, "REAL")] = pred

        df = data.copy()
        df.pred = y
        return df


    def feature_importances(self):
        """Used to get all feature importances of internal classifiers.
           Supplied models must support model.feature_importances_

        :return: DataFrame. Feature importances of internal classifiers.
        """
        df = pandas.DataFrame(index=self.columns, columns=list(self.machines.keys()))
        df.index.set_names(["feature_name", "day"], inplace=True)
        for attr, machine in self.machines.items():
            try:
                df[attr] = machine.feature_importances_
            except TypeError:
                raise FeatureImportanceMissingError()
        return df

    @staticmethod
    def f1(predicted_data):
        """

        :param predicted_data: LabeledData with filled out LabeledData.label and
               LabaledData.pred
        :return: Series with F1 score of all possible classes.
        """
        p = {}
        r = {}
        f1_score = {}
        columns = [(column, d["values"].keys()) for column, d in LABEL_GLOBAL.items() if d["train"]]
        real_columns = []
        for problem in PROBLEMS.values():
            prefix = f"problem_{problem}_"
            columns += [(prefix + attr, d["values"].keys()) for attr, d in LABEL_PROBLEM.items() if d["train"]]
            real_columns += [prefix + attr for attr, d in LABEL_PROBLEM_REAL.items() if d["train"]]
        for column, classes in columns:
            for cat in classes:
                idx = (column, cat)
                cat = str(cat)
                truth = predicted_data.label[column].replace(np.nan, _NONE, regex=True).values.astype("U")
                y = predicted_data.pred[column].replace(np.nan, _NONE, regex=True).values.astype("U")
                if np.any(np.char.equal(y, cat)):
                    p[idx] = np.sum(np.char.equal(truth, cat) * np.char.equal(truth, y)) / np.sum(np.char.equal(y, cat))
                else:
                    p[idx] = 0
                if np.any(np.char.equal(truth, cat)):
                    r[idx] = np.sum(np.char.equal(y, cat) * np.char.equal(truth, y)) / np.sum(np.char.equal(truth, cat))
                else:
                    r[idx] = 0
                if p[idx] + r[idx]:
                    f1_score[idx] = 2 * p[idx] * r[idx] / (p[idx] + r[idx])
                else:
                    f1_score[idx] = 0

        diff = predicted_data.pred[real_columns] - predicted_data.label[real_columns]
        rmse = np.sqrt(np.sum(np.square(diff)) / predicted_data.pred.shape[0]).transpose()

        df = pandas.DataFrame({"f1": f1_score, "precision": p, "recall": r})
        df.index.set_names(["property", "class"], inplace=True)
        df["rmse"] = rmse
        return df


class Error(Exception):
    pass


class AlreadyFittedError(Error):
    pass


class NotFittedError(Error):
    pass


class FeatureImportanceMissingError(Error):
    pass
