"""XGBoost en GPU y SHAP en GPU. Responsable: María José Yee.

Mismos hiperparámetros que la versión CPU, solo cambia device="cuda".
Datos con xgboost.QuantileDMatrix (o arreglos CuPy) para reducir memoria.
SHAP: booster.predict(dmatrix, pred_contribs=True) en GPU.
"""

# TODO(María José): entrenar_xgboost_gpu(X, y, params), shap_gpu(booster, X)
