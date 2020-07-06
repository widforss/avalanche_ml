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

class BulletinMachine:
    def __init__(self, ml_class_creator, ml_multi_creator, ml_real_creator):
        """Facilitates training and prediction of avalanche warnings.

        :param ml_class_creator: fn(in_size: Int, out_size: Int) -> classifier, where model supports model.fit(X, y)
                                 and model.predict(X) that returns softmax array or 1-hot.
        :param ml_multi_creator: fn(in_size: Int, out_size: Int) -> classifier, where model supports
                                 model.fit(X, y) and model.predict(X) that returns k-hot.
        :param ml_real_creator: fn(in_size: Int, out_size: Int) -> regressor, where model supports model.fit(X, y)
                                 and model.predict(X) that returns k-hot.
        """
        self.ml_class_creator = ml_class_creator
        self.ml_multi_creator = ml_multi_creator
        self.ml_real_creator = ml_real_creator
        self.machines_class = {}
        self.machines_multi = {}
        self.machines_real = {}
        self.labels_class = {}
        self.labels_multi = {}
        self.labels_real = {}
        self.X = None
        self.y = None
        self.is_timeseries = False
        self.fitted = False

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
        self.y = labeled_data.label
        dummies = labeled_data.to_dummies()
        self.dummies = dummies
        self.X = labeled_data.data
        prob_cols =\
            self.y.loc[:, [name.startswith("problem_") for name in self.y.columns.get_level_values(2)]]["CLASS", ""]
        for subprob, dummy in dummies['label']["CLASS"].items():
            if subprob == "":
                idx = [True] * dummy.shape[0]
            else:
                idx = np.any(np.char.equal(prob_cols.values.astype("U"), subprob), axis=1)
            try:
                self.machines_class[subprob] = self.ml_class_creator(self.X.shape[1:], len(dummy.columns))
            except ValueError:
                self.X = labeled_data.to_timeseries()[0]
                self.is_timeseries = True
                self.machines_class[subprob] = self.ml_class_creator(self.X.shape[1:], len(dummy.columns))
            print(f"Training class: {subprob}")
            if np.sum(idx):
                try:
                    self.machines_class[subprob].fit(self.X.loc[idx], dummy.loc[idx], epochs=epochs, verbose=verbose)
                except TypeError:
                    self.machines_class[subprob].fit(self.X.loc[idx], dummy.loc[idx])
        for subprob, dummy in dummies['label']["MULTI"].items():
            idx = np.any(np.char.equal(prob_cols.values.astype("U"), subprob), axis=1)
            self.machines_multi[subprob] = self.ml_multi_creator(self.X.shape[1:], len(dummy.columns))
            print(f"Training multi: {subprob}")
            if np.sum(idx):
                try:
                    self.machines_multi[subprob].fit(self.X.loc[idx], dummy.loc[idx], epochs=epochs, verbose=verbose)
                except TypeError:
                    self.machines_multi[subprob].fit(self.X.loc[idx], dummy.loc[idx])
        for subprob, values in dummies['label']["REAL"].items():
            idx = np.any(np.char.equal(prob_cols.values.astype("U"), subprob), axis=1)
            self.machines_real[subprob] = self.ml_real_creator(self.X.shape[1:], len(dummy.columns))
            print(f"Training real: {subprob}")
            if np.sum(idx):
                try:
                    self.machines_real[subprob].fit(self.X.loc[idx], values.loc[idx], epochs=epochs, verbose=verbose)
                except TypeError:
                    self.machines_real[subprob].fit(self.X.loc[idx], values.loc[idx])

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

        prim_dummy = self.dummies["label"]["CLASS"][""]
        idx = pandas.MultiIndex.from_arrays([
            prim_dummy.columns.get_level_values(2), prim_dummy.columns.get_level_values(3)
        ])
        prim_smax = pandas.DataFrame(self.machines_class[""].predict(X), columns=idx)
        for attr in ["danger_level", "emergency_warning"]:
            idx = np.argmax(prim_smax.loc[:, attr].values, axis=1)
            y["CLASS", "", attr] = np.array(list(prim_smax.loc[:, attr].columns))[idx]

        # Fix problem prediction to become a valid forecast.
        prob_cols = np.array([_NONE] + list(dict.fromkeys(PROBLEMS.values())))
        prob_smax = [pandas.DataFrame(prim_smax[f"problem_{n}"], columns=prob_cols).fillna(0) for n in [1, 2, 3]]
        prob_smax = np.stack(prob_smax, axis=1)
        idxs = np.flip(prob_smax.argsort(axis=2), axis=2)
        is_prob = np.any(np.sum(prob_smax.astype(np.bool), axis=2) > 1)
        for _ in [0, 1]:
            if is_prob:
                # If second likeliest problem_n-1 is very likely, use that instead
                fst = np.expand_dims(np.arange(y.shape[0]), axis=1)
                sec = [[0, 1]] * y.shape[0]
                likely = prob_smax[fst, sec, idxs[fst, sec, 1]] > 0.75 * prob_smax[fst, sec, idxs[fst, sec, 0]]
                idxs[:, 1:, 0] = idxs[:, 1:, 0] * np.invert(likely) + idxs[:, :-1, 1] * likely
                # If equal to problem_n-1/2, set to second likeliest alternative.
                prev_eq = idxs[:, 1:, :1] == idxs[:, :-1, :1]
                idxs[:, 1:, :-1] = idxs[:, 1:, :-1] * np.invert(prev_eq) + idxs[:, 1:, 1:] * prev_eq
            else:
                # If equal to problem_n-1/2, set to _NONE (as we only have 1-hot results).
                prev_eq = idxs[:, 1:, :1] == idxs[:, :-1, :1]
                idxs[:, 1:, :-1] = idxs[:, 1:, :-1] * np.invert(prev_eq)
            # Set to None if problem_n-1/2 was None.
            idxs[:, 1:] = idxs[:, 1:] * idxs[:, :-1, :1].astype(np.bool)
        for n in [1, 2, 3]:
            y["CLASS", "", f"problem_{n}"] = prob_cols[idxs[:, n - 1, 0]]
        y["CLASS", "", "problem_amount"] = np.sum(idxs[:, :, 0].astype(np.bool), axis=1).astype(np.int).astype("U")

        problem_cols = y.loc[:, [name.startswith("problem_") for name in y.columns.get_level_values(2)]]["CLASS", ""]
        for subprob, dummy in self.dummies['label']["CLASS"].items():
            if subprob == _NONE:
                continue
            rows = np.any(np.char.equal(problem_cols.values.astype("U"), subprob), axis=1)
            if np.sum(rows):
                dummy = self.dummies['label']["CLASS"][subprob]
                pred = pandas.DataFrame(self.machines_class[subprob].predict(X[np.ix_(rows)]), columns=dummy.columns)
                for attr in pred.columns.get_level_values(2).unique():
                    idx = np.argmax(pred["CLASS", subprob].loc[:, attr].values, axis=1)
                    labels = np.array(list(pred["CLASS", subprob].loc[:, attr].columns))[idx]
                    y["CLASS", subprob, attr].values[np.ix_(rows)] = labels
        for subprob, dummy in self.dummies['label']["MULTI"].items():
            rows = np.any(np.char.equal(problem_cols.values.astype("U"), subprob), axis=1)
            if np.sum(rows):
                dummy = self.dummies['label']["MULTI"][subprob]
                pred = pandas.DataFrame(self.machines_multi[subprob].predict(X[np.ix_(rows)]), columns=dummy.columns)
                for attr in pred.columns.get_level_values(2).unique():
                    labels = pred["MULTI", subprob, attr].values.astype(np.int).astype("U")
                    y["MULTI", subprob, attr] = "0"
                    y["MULTI", subprob, attr].values[np.ix_(rows)] = [''.join(row) for row in labels]
        for subprob, dummy in self.dummies['label']["REAL"].items():
            rows = np.any(np.char.equal(problem_cols.values.astype("U"), subprob), axis=1)
            if np.sum(rows):
                dummy = self.dummies['label']["REAL"][subprob]
                pred = pandas.DataFrame(self.machines_real[subprob].predict(X[np.ix_(rows)]), columns=dummy.columns)
                for attr in pred.columns.get_level_values(2).unique():
                    y["REAL", subprob, attr] = 0
                    y["REAL", subprob, attr].values[np.ix_(rows)] = pred["REAL", subprob, attr].values

        df = data.copy()
        df.pred = y
        return df


    def feature_importances(self):
        """Used to get all feature importances of internal classifiers.
           Supplied models must support model.feature_importances_

        :return: DataFrame. Feature importances of internal classifiers.
        """
        importances = {}
        tupls = list(self.machines_class.items()) + list(self.machines_multi.items()) + list(self.machines_real.items())
        for attr, machine in self.machines_class.items():
            try:
                importances[('CLASS', attr)] = machine.feature_importances_
            except AttributeError:
                pass
        for attr, machine in self.machines_multi.items():
            try:
                importances[('MULTI', attr)] = machine.feature_importances_
            except AttributeError:
                pass
        for attr, machine in self.machines_real.items():
            try:
                importances[('REAL', attr)] = machine.feature_importances_
            except AttributeError:
                pass
        df = pandas.DataFrame(importances, index=self.X.columns)
        df.index.set_names(["feature_name", "day"], inplace=True)
        return df


class Error(Exception):
    pass


class AlreadyFittedError(Error):
    pass


class NotFittedError(Error):
    pass


class FeatureImportanceMissingError(Error):
    pass
