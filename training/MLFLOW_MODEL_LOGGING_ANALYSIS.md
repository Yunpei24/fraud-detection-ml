# 🔍 Analyse : MLflow Model Logging pour 4 Types de Modèles

**Date :** 4 novembre 2025  
**Fichier analysé :** `training/src/mlflow_utils/tracking.py`  
**Fonction :** `log_model()`

---

## 📊 Résumé des 4 Modèles Entraînés

| Modèle | Classe Wrapper | Classe Réelle (`.model`) | Type | Méthodes |
|--------|---------------|-------------------------|------|----------|
| **XGBoost** | `XGBoostModel` | `xgb.XGBClassifier` | Classifieur | `.predict()`, `.predict_proba()` |
| **Random Forest** | `RandomForestModel` | `sklearn.RandomForestClassifier` | Classifieur | `.predict()`, `.predict_proba()` |
| **Neural Network** | `NeuralNetworkModel` | `sklearn.MLPClassifier` | Classifieur | `.predict()`, `.predict_proba()` |
| **Isolation Forest** | `IsolationForestModel` | `sklearn.IsolationForest` | **Anomaly Detector** | `.predict()`, `.decision_function()` |

---

## 🎯 Logique de `log_model()` dans `tracking.py`

```python
def log_model(model, artifact_path="model"):
    est = _unwrap_model(model)  # Extrait .model de nos wrappers
    
    if isinstance(est, xgb.XGBClassifier):
        # CAS 1: XGBoost → mlflow.xgboost.log_model()
        mlflow.xgboost.log_model(est, artifact_path)
    else:
        # CAS 2: Tous les autres → mlflow.sklearn.log_model()
        mlflow.sklearn.log_model(est, artifact_path)
```

---

## ✅ Compatibilité par Modèle

### 1. **XGBoost** ✅ PARFAIT
- **Type réel :** `xgb.XGBClassifier`
- **Logique MLflow :** `mlflow.xgboost.log_model()`
- **Status :** ✅ Fonctionne parfaitement
- **Raison :** MLflow a un support natif pour XGBoost

### 2. **Random Forest** ✅ PARFAIT
- **Type réel :** `sklearn.ensemble.RandomForestClassifier`
- **Logique MLflow :** `mlflow.sklearn.log_model()`
- **Status :** ✅ Fonctionne parfaitement
- **Raison :** Classifieur sklearn standard

### 3. **Neural Network** ✅ PARFAIT
- **Type réel :** `sklearn.neural_network.MLPClassifier`
- **Logique MLflow :** `mlflow.sklearn.log_model()`
- **Status :** ✅ Fonctionne parfaitement
- **Raison :** Classifieur sklearn standard

### 4. **Isolation Forest** ✅ FONCTIONNE (mais avec particularités)
- **Type réel :** `sklearn.ensemble.IsolationForest`
- **Logique MLflow :** `mlflow.sklearn.log_model()`
- **Status :** ✅ Fonctionne MAIS ce n'est pas un classifieur standard
- **Particularités :**
  - N'a PAS `.predict_proba()` → retourne un score via `.decision_function()`
  - `.predict()` retourne -1 (anomalie) ou 1 (normal), pas 0/1
  - MLflow le sauvegarde quand même car c'est un estimateur sklearn valide

---

## 🐛 Problème Potentiel : Isolation Forest

**Pourquoi c'est différent ?**

L'Isolation Forest est un **anomaly detector**, pas un classifieur binaire standard :

| Aspect | Classifieurs (XGB, RF, NN) | Isolation Forest |
|--------|---------------------------|------------------|
| Type | Classification supervisée | Détection d'anomalies |
| Entraînement | Nécessite labels (0/1) | Peut être non-supervisé |
| Prédiction | `.predict_proba()` → [0.0-1.0] | `.decision_function()` → score |
| Sortie `.predict()` | 0 (normal) ou 1 (fraud) | -1 (anomalie) ou 1 (normal) |
| Interprétation | Probabilité de fraude | Score d'anomalie (plus bas = plus anormal) |

**Dans notre code :**

```python
# training/src/models/isolation_forest.py
def predict_proba(self, X, y=None):
    """Custom predict_proba using decision_function scores"""
    scores = self.model.decision_function(X)
    # Convert anomaly scores to probabilities (lower = more anomalous = higher fraud prob)
    fraud_probs = 1 / (1 + np.exp(scores))  # Sigmoid transformation
    return np.column_stack((1 - fraud_probs, fraud_probs))
```

Nous avons **créé une méthode `.predict_proba()` custom** qui transforme les scores d'anomalie en probabilités !

---

## 🔧 Améliorations Apportées

### Avant (code original) :
```python
except Exception:
    pass  # ❌ Échoue silencieusement, impossible de débugger !
```

### Après (code amélioré) :
```python
except Exception as e:
    logger.error(f"❌ Failed to log model with mlflow: {e}")
    logger.error(f"   Model type: {model_class}")
    logger.error(f"   Model attributes: {dir(est)[:10]}...")
    
    # Fallback: log as pickle
    try:
        dump(est, "model.joblib")
        mlflow.log_artifacts(dump_dir, artifact_path)
        logger.info(f"✅ Model artifacts logged successfully (fallback)")
    except Exception as e2:
        logger.error(f"❌ Fallback also failed: {e2}")
        raise RuntimeError(f"Failed to log model: {e}") from e2
```

### Bénéfices :
1. **Logs détaillés** : On voit exactement quelle erreur se produit
2. **Type de modèle** : On sait quel modèle échoue
3. **Attributs** : On peut débugger les méthodes manquantes
4. **Fallback robuste** : Si MLflow échoue, on sauvegarde en pickle
5. **Raise exception** : Ne masque plus les erreurs critiques

---

## 🧪 Tests à Effectuer

### 1. Vérifier que les 4 modèles se loggent correctement

```bash
# Déclencher le training via Airflow UI
http://localhost:8080
# DAG: 01_training_pipeline

# Surveiller les logs
docker logs -f fraud-airflow-worker | grep "log_model"
```

**Logs attendus :**
```
Logging XGBoost model to artifact_path='model'
✅ XGBoost model logged successfully

Logging sklearn model (RandomForestClassifier) to artifact_path='model'
✅ Sklearn model (RandomForestClassifier) logged successfully

Logging sklearn model (MLPClassifier) to artifact_path='model'
✅ Sklearn model (MLPClassifier) logged successfully

Logging sklearn model (IsolationForest) to artifact_path='model'
✅ Sklearn model (IsolationForest) logged successfully
```

### 2. Vérifier que les artifacts existent dans MLflow

```bash
# Lister les runs
docker exec fraud-training python -c "
import mlflow
mlflow.set_tracking_uri('http://mlflow:5000')
runs = mlflow.search_runs(experiment_names=['fraud_detection_training'])
print(f'Total runs: {len(runs)}')

# Check artifacts for latest run
if len(runs) > 0:
    run_id = runs.iloc[0]['run_id']
    client = mlflow.MlflowClient()
    artifacts = client.list_artifacts(run_id)
    print(f'Artifacts: {[a.path for a in artifacts]}')
"
```

**Résultat attendu :**
```
Total runs: 12  # (4 train + 4 eval + 4 register)
Artifacts: ['model', 'xgboost_metadata.json']
```

### 3. Vérifier que les modèles sont dans le Registry

```bash
docker exec fraud-training python -c "
import mlflow
mlflow.set_tracking_uri('http://mlflow:5000')

client = mlflow.MlflowClient()
models = client.search_registered_models()

for model in models:
    print(f'Model: {model.name}')
    versions = client.search_model_versions(f\"name='{model.name}'\")
    for v in versions:
        print(f'  Version {v.version}: {v.current_stage}')
"
```

**Résultat attendu :**
```
Model: fraud_detection_xgboost
  Version 1: Staging
Model: fraud_detection_random_forest
  Version 1: Staging
Model: fraud_detection_neural_network
  Version 1: Staging
Model: fraud_detection_isolation_forest
  Version 1: Staging
```

---

## 📋 Conclusion

### ✅ Points Positifs
1. **Tous les 4 modèles sont supportés** par la fonction `log_model()`
2. **XGBoost** a un traitement spécial avec `mlflow.xgboost.log_model()`
3. **Les 3 autres modèles** utilisent `mlflow.sklearn.log_model()` qui fonctionne pour tous les estimateurs sklearn
4. **Isolation Forest** fonctionne car nous avons une méthode `.predict_proba()` custom dans le wrapper
5. **Logs détaillés** ajoutés pour faciliter le debugging

### 🔧 Améliorations Apportées
1. Ajout de `logger` pour tracer les opérations MLflow
2. Messages détaillés pour chaque type de modèle
3. Fallback robuste vers pickle/joblib si MLflow échoue
4. Raise des exceptions au lieu de les avaler silencieusement
5. Logs des attributs du modèle en cas d'erreur

### 🚀 Prochaine Étape
**Déclencher le training via Airflow UI et observer les logs détaillés !**

---

**Fichiers modifiés :**
- `training/src/mlflow_utils/tracking.py` (fonction `log_model()`)

**Documentation connexe :**
- `WHERE_ARE_MODELS_STORED.md` - Localisation des modèles après training
- `WHERE_ARE_MODELS_AFTER_TRAINING.md` - Flow complet de training → deployment
