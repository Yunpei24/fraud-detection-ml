from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from training.src.data.utils import fill_na
from training.src.features.engineer import build_feature_frame
from training.src.data.splitter import stratified_split
from training.src.models.random_forest import RandomForestModel
from training.src.evaluation.metrics import calculate_all_metrics
from sklearn.model_selection import train_test_split

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

def test_end_to_end_pipeline(tiny_credit_df):
    # Processing pipeline
    df = fill_na(tiny_credit_df)
    feats = build_feature_frame(df)
    X = feats.drop(columns=["Class"])
    y = df["Class"].astype(int).values

    # Check if any class has less than 2 samples
    unique_classes, class_counts = np.unique(y, return_counts=True)
    print(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
    
    # Ensure that both classes have more than 1 sample
    if np.min(class_counts) < 2:
        raise ValueError("At least one class has fewer than 2 samples. Stratified split cannot be performed.")
    
    # Apply SMOTE to oversample the minority class
    smote = SMOTE(random_state=42, k_neighbors=2)  # Reduced n_neighbors to 2
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Now you can perform a stratified split without issues
    Xtr, Xte, ytr, yte = train_test_split(X_resampled, y_resampled, test_size=0.25, random_state=42)

    # Initialize and train model
    model = RandomForestModel(params={"n_estimators": 10, "max_depth": 3, "n_jobs": 1})
    model.train(Xtr, ytr, Xte, yte)

    # Predicting probabilities and classes
    proba = model.predict_proba(Xte)
    ypred = (proba >= 0.5).astype(int)

    # Evaluate model performance
    metr = calculate_all_metrics(yte, proba, ypred)

    # Assert metrics values
    for k in ["auc", "pr_auc", "precision", "recall", "f1"]:
        assert k in metr
        assert 0.0 <= metr[k] <= 1.0
