"""Entrenamientos de referencia en CPU. Responsable: María José Yee.

    Regresión Logística : scikit-learn (StandardScaler + LogisticRegression,
                          class_weight="balanced", C en {0.1, 1, 10})
    XGBoost             : tree_method="hist", device="cpu",
                          n_jobs=1 (secuencial) y n_jobs=-1 (multinúcleo)
"""

# TODO(María José): entrenar_logistica_cpu(X, y, C, n_jobs)
# TODO(María José): entrenar_xgboost_cpu(X, y, params, n_jobs)
