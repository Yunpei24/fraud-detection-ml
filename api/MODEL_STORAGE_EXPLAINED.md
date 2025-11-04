# 📦 Où sont stockés les modèles ML ?

**Date :** 4 novembre 2025  
**Container :** `fraud-api`

---

## 🎯 Résumé de la situation actuelle

### ✅ **Configuration :**
- **Chemin configuré :** `/mnt/fraud-models/champion/`
- **État actuel :** ❌ Aucun modèle réel n'existe
- **Fallback activé :** ✅ L'API utilise des **modèles mock** (factices)

---

## 📂 Structure de stockage des modèles

### **1. Configuration du chemin dans l'API**

```python
# api/src/config/settings.py
model_path = os.getenv(
    "ML_MODEL_PATH",
    os.getenv("MODEL_PATH", os.getenv("AZURE_STORAGE_MOUNT_PATH", "/mnt/fraud-models")),
)
```

**Priorité de résolution :**
1. `ML_MODEL_PATH` (variable d'environnement)
2. `MODEL_PATH` (variable d'environnement)
3. `AZURE_STORAGE_MOUNT_PATH` (variable d'environnement)
4. `/mnt/fraud-models` (défaut)

### **2. Chemin actuel dans le container**

```bash
$ docker exec fraud-api python -c "from src.config.settings import settings; print(settings.model_path)"
/mnt/fraud-models
```

### **3. Répertoire complet avec traffic routing**

Le système utilise un système de **champion/canary** :

```
/mnt/fraud-models/
├── champion/              # Modèles de production (100% du trafic)
│   ├── xgboost_fraud_model.pkl
│   ├── random_forest_fraud_model.pkl
│   ├── nn_fraud_model.pth
│   ├── isolation_forest_model.pkl
│   ├── shap_explainer_xgb.pkl
│   ├── shap_explainer_rf.pkl
│   ├── shap_explainer_nn.pkl
│   └── shap_explainer_iforest.pkl
│
└── canary/               # Modèles en test (0-25% du trafic)
    ├── xgboost_fraud_model.pkl
    ├── random_forest_fraud_model.pkl
    ├── nn_fraud_model.pth
    └── isolation_forest_model.pkl
```

---

## 🔍 État actuel dans le container

### **Vérification manuelle :**

```bash
# 1. Vérifier si le répertoire existe
$ docker exec fraud-api ls -lah /mnt/fraud-models/
ls: cannot access '/mnt/fraud-models/': No such file or directory
```

**❌ Le répertoire n'existe pas !**

### **Logs du container :**

```json
{
  "level": "WARNING",
  "message": "Isolation Forest not found at /mnt/fraud-models/champion/isolation_forest_model.pkl, using mock"
}
{
  "level": "WARNING",
  "message": "SHAP explainer (XGBoost) not found at /mnt/fraud-models/champion/shap_explainer_xgb.pkl"
}
{
  "level": "INFO",
  "message": "All models loaded successfully"
}
{
  "level": "INFO",
  "message": "Available models: ['xgboost', 'random_forest', 'neural_network', 'isolation_forest', 'ensemble']"
}
```

**✅ L'API fonctionne avec des modèles mock (factices) !**

---

## 🤖 Système de modèles Mock

L'API a un mécanisme de fallback qui crée des **modèles factices** quand les vrais modèles n'existent pas :

### **Code de fallback :**

```python
# api/src/models/ml_models/ensemble.py

def load_models(self) -> None:
    """Load all models from disk."""
    
    # Essayer de charger XGBoost
    xgboost_path = os.path.join(self.models_path, settings.xgboost_model_name)
    if os.path.exists(xgboost_path):
        with open(xgboost_path, "rb") as f:
            self.xgboost_model = pickle.load(f)
        logger.info("✅ XGBoost model loaded")
    else:
        logger.warning(f"XGBoost model not found at {xgboost_path}, using mock")
        self.xgboost_model = self._create_mock_model("xgboost")  # ← Mock !
```

### **Modèles mock actuellement actifs :**

| Modèle | Fichier attendu | État | Type utilisé |
|--------|----------------|------|--------------|
| XGBoost | `xgboost_fraud_model.pkl` | ❌ Non trouvé | 🤖 Mock |
| Random Forest | `random_forest_fraud_model.pkl` | ❌ Non trouvé | 🤖 Mock |
| Neural Network | `nn_fraud_model.pth` | ❌ Non trouvé | 🤖 Mock |
| Isolation Forest | `isolation_forest_model.pkl` | ❌ Non trouvé | 🤖 Mock |
| SHAP Explainers | `shap_explainer_*.pkl` | ❌ Non trouvés | ❌ Désactivés |

---

## 📍 Où sont créés les vrais modèles ?

### **1. Container de training (`fraud-training`)**

Les modèles sont créés par le **DAG Airflow `01_training_pipeline`** :

```bash
# Dans le container training
/app/models/              # Modèles sauvegardés localement
/mlflow/artifacts/        # Modèles enregistrés dans MLflow
```

**Commande pour vérifier :**
```bash
docker exec fraud-training ls -lah /app/models/
```

### **2. MLflow Model Registry**

Les modèles entraînés sont **enregistrés dans MLflow** :

- **URL MLflow :** http://localhost:5001
- **Registry path :** `/mlflow/artifacts/`
- **Stages :** None → Staging → Production

**Voir les modèles dans MLflow :**
```bash
curl http://localhost:5001/api/2.0/mlflow/registered-models/list | jq .
```

### **3. Azure File Share (Production uniquement)**

En production sur Azure, les modèles sont stockés dans **Azure File Share** :

- **Storage Account :** `joshfraudstorageaccount`
- **File Share :** `fraud-models`
- **Mount point :** `/mnt/fraud-models`

---

## 🔄 Comment les modèles arrivent dans l'API ?

### **Flow complet :**

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING → API DEPLOYMENT                     │
└─────────────────────────────────────────────────────────────────┘

ÉTAPE 1 : ENTRAÎNEMENT
├─ Airflow DAG 01_training_pipeline
├─ Container fraud-training
├─ Entraîne XGBoost, RF, NN, Isolation Forest
├─ Sauvegarde dans /app/models/
└─ Enregistre dans MLflow Registry → Stage: None

ÉTAPE 2 : PROMOTION STAGING
├─ Airflow DAG 05_model_deployment_canary_http
├─ Promeut models: None → Staging dans MLflow
├─ Script deploy_canary.py
│  ├─ Télécharge models depuis MLflow
│  ├─ Sauvegarde dans /mnt/fraud-models/canary/
│  └─ Met à jour traffic_routing.json (5% canary)
└─ API auto-reload détecte les nouveaux fichiers

ÉTAPE 3 : CANARY 25%
├─ Airflow DAG 05_model_deployment_canary_http
├─ Met à jour traffic_routing.json (25% canary)
└─ API auto-reload détecte le changement

ÉTAPE 4 : PROMOTION PRODUCTION
├─ Airflow DAG 05_model_deployment_canary_http
├─ Promeut models: Staging → Production dans MLflow
├─ Script promote_to_production.py
│  ├─ Copie /mnt/fraud-models/canary/ → /mnt/fraud-models/champion/
│  └─ Met à jour traffic_routing.json (canary disabled)
└─ API auto-reload détecte les nouveaux fichiers
```

---

## 🛠️ Comment créer les modèles manuellement ?

### **Méthode 1 : Déclencher le DAG de training**

```bash
# 1. Aller dans Airflow UI
http://localhost:8080

# 2. Trouver le DAG "01_training_pipeline"

# 3. Cliquer sur "Trigger DAG"

# 4. Attendre la fin de l'entraînement (~30-60 minutes)

# 5. Vérifier les modèles dans MLflow
http://localhost:5001
```

### **Méthode 2 : Entraînement manuel dans le container**

```bash
# 1. Entrer dans le container training
docker exec -it fraud-training bash

# 2. Lancer le script de training
python -m src.pipelines.training_pipeline

# 3. Vérifier les modèles créés
ls -lah /app/models/

# 4. Copier vers l'API (temporaire pour dev)
docker cp fraud-training:/app/models/xgboost_fraud_model.pkl /tmp/
docker exec fraud-api mkdir -p /mnt/fraud-models/champion
docker cp /tmp/xgboost_fraud_model.pkl fraud-api:/mnt/fraud-models/champion/
```

### **Méthode 3 : Utiliser des modèles de test**

Pour le développement local, vous pouvez créer des modèles simples :

```python
# Dans le container API
docker exec -it fraud-api python

>>> import pickle
>>> from sklearn.ensemble import RandomForestClassifier
>>> import os
>>> 
>>> # Créer le répertoire
>>> os.makedirs("/mnt/fraud-models/champion", exist_ok=True)
>>> 
>>> # Créer un modèle simple
>>> model = RandomForestClassifier(n_estimators=10)
>>> 
>>> # Sauvegarder
>>> with open("/mnt/fraud-models/champion/xgboost_fraud_model.pkl", "wb") as f:
...     pickle.dump(model, f)
>>> 
>>> print("✅ Modèle de test créé !")
```

---

## 🔍 Commandes de diagnostic

### **1. Vérifier le chemin configuré**

```bash
docker exec fraud-api python -c "from src.config.settings import settings; print('Model Path:', settings.model_path)"
```

### **2. Lister les modèles disponibles**

```bash
docker exec fraud-api find /mnt/fraud-models -name "*.pkl" -o -name "*.pth"
```

### **3. Vérifier les logs de chargement**

```bash
docker logs fraud-api 2>&1 | grep -i "model\|loading"
```

### **4. Tester l'API avec modèles mock**

```bash
# Obtenir un token
TOKEN=$(curl -s -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin123" | jq -r '.access_token')

# Lister les modèles
curl -X GET "http://localhost:8000/api/v1/explain/models" \
  -H "Authorization: Bearer $TOKEN"

# Résultat avec modèles mock :
["xgboost", "random_forest", "neural_network", "isolation_forest", "ensemble"]
```

### **5. Vérifier le status des volumes Docker**

```bash
docker volume ls | grep fraud
docker volume inspect fraud-detection-ml_mlflow_artifacts
```

---

## 📋 Résumé

| Question | Réponse |
|----------|---------|
| **Où sont stockés les modèles ?** | `/mnt/fraud-models/champion/` (configuré) |
| **Les modèles existent-ils actuellement ?** | ❌ Non, le répertoire n'existe pas |
| **L'API fonctionne quand même ?** | ✅ Oui, avec des modèles mock (factices) |
| **Comment créer les vrais modèles ?** | Déclencher DAG Airflow `01_training_pipeline` |
| **Où sont les modèles après training ?** | `/mlflow/artifacts/` dans MLflow Registry |
| **Comment les déployer dans l'API ?** | Via DAG `05_model_deployment_canary_http` |
| **Peut-on tester sans vrais modèles ?** | ✅ Oui, les modèles mock permettent de tester l'API |

---

## 🚀 Prochaines étapes

1. **Lancer le training** pour créer les vrais modèles
2. **Enregistrer dans MLflow** pour versioning
3. **Déployer via DAG canary** pour production-ready
4. **Tester avec vrais modèles** pour validation complète

---

**Besoin d'aide ?** Consultez :
- [AUTO_RELOAD_GUIDE.md](AUTO_RELOAD_GUIDE.md) - Auto-reload des modèles
- [DEPLOYMENT_API.md](DEPLOYMENT_API.md) - Déploiement canary
- [README.md](../README.md) - Documentation générale
