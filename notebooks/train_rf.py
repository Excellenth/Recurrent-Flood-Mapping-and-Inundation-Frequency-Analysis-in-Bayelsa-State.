"""Train and evaluate the RF flood classifier from the locally-saved training samples.

Does two things with the same merged sample set:
1. A local scikit-learn RF for a fast, reliable accuracy check (train/test split,
   confusion matrix, feature importances).
2. An ee.Classifier.smileRandomForest trained on the same data, for classifying full
   images server-side in GEE (needed for Phase 2's inundation frequency mapping).
"""

import sys
import json

sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, cohen_kappa_score

import ee
from gee_utils import init_ee
from classification import FEATURE_BANDS

records = []
for year in ["2019", "2020", "2022"]:
    with open(f"data/training_samples_{year}.json") as fh:
        records.extend(json.load(fh))

df = pd.DataFrame(records)
print("Total samples:", len(df))
print("Class balance:\n", df["label"].value_counts())
print("By event year:\n", df.groupby(["event_year", "label"]).size())

X = df[FEATURE_BANDS]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("\n--- scikit-learn RF (local, held-out test split) ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Cohen's Kappa:", cohen_kappa_score(y_test, y_pred))
print("Confusion matrix (rows=true, cols=pred, [0,1]):\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["non-flood", "flood"]))

importances = pd.Series(rf.feature_importances_, index=FEATURE_BANDS).sort_values(ascending=False)
print("Feature importances:\n", importances)

# --- Train the ee.Classifier version for server-side use ---
init_ee()
features = [ee.Feature(None, r) for r in records]
fc = ee.FeatureCollection(features)
ee_classifier = ee.Classifier.smileRandomForest(numberOfTrees=200, seed=42).train(
    features=fc, classProperty="label", inputProperties=FEATURE_BANDS
)
print("\nee.Classifier trained on", fc.size().getInfo(), "samples.")

# Quick server-side resubstitution check (small FC, should be cheap/fast)
train_matrix = fc.classify(ee_classifier).errorMatrix("label", "classification")
print("ee.Classifier resubstitution accuracy:", train_matrix.accuracy().getInfo())
