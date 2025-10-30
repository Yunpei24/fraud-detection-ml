from src.features.engineer import add_behavioral_features, build_feature_frame  # Corrected import path

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

def test_add_behavioral_features_creates_cols(tiny_credit_df):
    out = add_behavioral_features(tiny_credit_df)
    # Correct the column name to 'amt_rollmean_10' instead of 'amt_rolling_mean_10'
    assert "amt_rollmean_10" in out.columns
    assert "amt_z" in out.columns

def test_build_feature_frame_smoke(tiny_credit_df):
    out = build_feature_frame(tiny_credit_df, ts_col=None, country_col=None)
    assert out.shape[0] == tiny_credit_df.shape[0]
