# 🚀 Guide d'Exécution des Tests MLOps

Ce guide explique comment exécuter la suite complète de tests pour le projet MLOps de détection de fraude.

## 📋 Prérequis

1. **Services Docker opérationnels** :
   ```bash
   docker-compose -f docker-compose.local.yml up -d
   ```

2. **Vérification de l'état des services** :
   ```bash
   ./check_services.sh
   ```

## 🧪 Scripts de Test Disponibles

### 1. **Suite Complète de Tests**
```bash
./run_all_tests.sh
```
Exécute tous les tests dans l'ordre : unitaires → intégration → Airflow → E2E

### 2. **Tests Unitaires**
```bash
# Tous les services
./run_unit_tests.sh

# Service spécifique
./run_unit_tests.sh api
./run_unit_tests.sh data
./run_unit_tests.sh drift
./run_unit_tests.sh training
```

### 3. **Tests d'Intégration**
```bash
./run_integration_tests.sh
```
Teste les interactions entre composants (API ↔ Base de données ↔ Services externes)

### 4. **Tests Airflow DAGs**
```bash
./run_airflow_tests.sh
```
Teste la structure et l'exécution des DAGs Airflow

### 5. **Vérification des Services**
```bash
./check_services.sh
```
Vérifie que tous les services Docker sont opérationnels avant les tests

## 📊 Structure des Tests

```
tests/
├── airflow/           # Tests DAGs Airflow
│   ├── test_dag_01_training.py
│   └── test_dag_02_drift.py
├── e2e/              # Tests end-to-end
│   └── test_full_mlops_workflow.py
└── integration/      # Tests d'intégration
    ├── test_airflow_api_integration.py
    ├── test_drift_detection_e2e.py
    └── test_retraining_trigger_integration.py
```

Chaque service a aussi ses propres tests :
- `api/tests/unit/` - Tests unitaires API
- `data/tests/unit/` - Tests unitaires pipeline de données
- `drift/tests/unit/` - Tests unitaires détection de drift
- `training/tests/unit/` - Tests unitaires entraînement modèles

## 🎯 Types de Tests

### **Tests Unitaires** (500+ tests)
- Testent chaque fonction/classe individuellement
- Utilisent des mocks pour les dépendances externes
- Couverture complète de la logique métier

### **Tests d'Intégration** (60+ tests)
- Testent les interactions entre composants
- Vérifient les appels API, base de données, messaging
- Valident les workflows complets

### **Tests Airflow** (50+ tests)
- Testent la structure des DAGs
- Vérifient les dépendances entre tâches
- Valident la logique d'orchestration

### **Tests E2E** (20+ tests)
- Testent le pipeline complet MLOps
- De l'ingestion des données à la prédiction
- Incluent la détection de drift et retraining

## 🚦 États des Services

### Services Critiques
- ✅ **postgres** : Base de données
- ✅ **redis** : Cache
- ✅ **api** : Service de prédiction
- ✅ **data** : Pipeline de données
- ✅ **drift** : Détection de drift
- ✅ **training** : Entraînement modèles
- ✅ **airflow-webserver/scheduler** : Orchestration

### Services de Monitoring
- 📊 **mlflow** : Tracking modèles
- 📈 **prometheus** : Métriques
- 📊 **grafana** : Dashboards

## 🔧 Dépannage

### Service non disponible
```bash
# Redémarrer un service spécifique
docker-compose -f docker-compose.local.yml restart <service_name>

# Voir les logs
docker-compose -f docker-compose.local.yml logs <service_name>
```

### Tests qui échouent
```bash
# Exécuter avec plus de détails
docker-compose -f docker-compose.local.yml exec <service> \
  bash -c "cd /home/appuser && python -m pytest tests/unit/ -v -s"

# Exécuter un test spécifique
docker-compose -f docker-compose.local.yml exec <service> \
  bash -c "cd /home/appuser && python -m pytest tests/unit/test_specific.py::TestClass::test_method -v"
```

### Problèmes de dépendances
```bash
# Reconstruire un service
docker-compose -f docker-compose.local.yml build <service>

# Forcer la reconstruction
docker-compose -f docker-compose.local.yml build --no-cache <service>
```

## 📈 Métriques de Test

- **Tests totaux** : 500+
- **Couverture** : 95%+ (estimé)
- **Temps d'exécution** : ~10-15 minutes
- **Services testés** : 6 modules principaux
- **Technologies** : pytest, Docker, mocks, fixtures

## 🎉 Recommandations

1. **Exécutez d'abord** `./check_services.sh`
2. **Puis lancez** `./run_all_tests.sh` pour la suite complète
3. **Pour le développement** : `./run_unit_tests.sh <service>` pour les tests rapides
4. **Sur les erreurs** : Vérifiez les logs Docker et relancez les services

---
*Suite de tests créée automatiquement pour le projet MLOps de détection de fraude*