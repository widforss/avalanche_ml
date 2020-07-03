from aggregatedata import ForecastDataset, LabeledData, REG_ENG, CsvMissingError
from machine import BulletinMachine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import BayesianRidge

def classifier_creator(indata, outdata):
    return RandomForestClassifier()

def regressor_creator(indata, outdata):
    return BayesianRidge()

days = 7
regobs_types = list(REG_ENG.keys())
print("Reading csv")
labeled_data = None
try:
    labeled_data = LabeledData.from_csv(days=days, regobs_types=regobs_types)
except CsvMissingError:
    labeled_data = ForecastDataset(regobs_types=regobs_types).label(days=days)
    labeled_data.to_csv()

f1 = None
importances = None
for split_idx, (training_data, testing_data) in enumerate(labeled_data.kfold(5)):
    print(f"Training fold: {split_idx}")
    bm = BulletinMachine(classifier_creator, classifier_creator, regressor_creator)
    bm.fit(training_data, epochs=80, verbose=1)

    print(f"Testing fold: {split_idx}")
    predicted_data = bm.predict(testing_data)
    labeled_data.pred.loc[predicted_data.pred.index] = predicted_data.pred
    split_imp = bm.feature_importances()
    importances = split_imp if importances is None else importances + (split_imp - importances) / (split_idx + 1)
    f1_series = bm.f1(predicted_data)
    f1 = f1_series if f1 is None else f1 + (f1_series - f1) / (split_idx + 1)
    break

print("Writing predictions")
predicted_data.pred.to_csv("pred.csv", sep=';')
print("Writing importances")
importances.to_csv("importances.csv", sep=';')
print("Writing F1 scores")
f1.to_csv("f1.csv", sep=";")
