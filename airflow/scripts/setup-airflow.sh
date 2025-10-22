#!/bin/bash

# Script de setup Airflow pour Fraud Detection
# Usage: ./setup-airflow.sh

set -e

echo "🚀 Setup Airflow pour Fraud Detection"
echo "======================================"

# Vérifier Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker n'est pas installé"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose n'est pas installé"
    exit 1
fi

echo "✅ Docker et Docker Compose installés"

# Vérifier .env
if [ ! -f .env ]; then
    echo "📝 Création du fichier .env depuis .env.example"
    cp .env.example .env
    echo "⚠️  IMPORTANT: Modifier .env avec vos credentials"
    echo "   - DATABRICKS_HOST"
    echo "   - DATABRICKS_TOKEN"
    echo "   - ALERT_EMAIL_RECIPIENTS"
else
    echo "✅ Fichier .env existe"
fi

# Créer les dossiers nécessaires
echo "📁 Création des dossiers"
mkdir -p logs dags plugins config scripts

# Set AIRFLOW_UID
echo "🔧 Configuration AIRFLOW_UID"
if [ -z "$AIRFLOW_UID" ]; then
    export AIRFLOW_UID=50000
    echo "export AIRFLOW_UID=50000" >> .env
fi
echo "✅ AIRFLOW_UID=$AIRFLOW_UID"

# Initialiser la base de données Airflow
echo "🗄️  Initialisation de la base de données Airflow"
docker-compose -f docker-compose.airflow.yml up airflow-init

# Démarrer les services
echo "🚀 Démarrage des services Airflow"
docker-compose -f docker-compose.airflow.yml up -d

# Attendre que les services soient prêts
echo "⏳ Attente que les services démarrent (30s)"
sleep 30

# Vérifier l'état des services
echo "🔍 Vérification des services"
docker-compose -f docker-compose.airflow.yml ps

# Vérifier les DAGs
echo "📊 Liste des DAGs"
docker exec -it airflow-scheduler airflow dags list || echo "⚠️  DAGs pas encore chargés"

echo ""
echo "✅ Setup terminé!"
echo ""
echo "📋 Prochaines étapes:"
echo "1. Accéder à Airflow UI: http://localhost:8080"
echo "   - Username: airflow"
echo "   - Password: airflow"
echo ""
echo "2. Activer les DAGs dans l'UI:"
echo "   - 02_drift_monitoring (CRITIQUE)"
echo "   - 01_training_pipeline"
echo ""
echo "3. Vérifier les logs:"
echo "   docker-compose -f docker-compose.airflow.yml logs -f airflow-scheduler"
echo ""
echo "4. Configurer les connexions Airflow:"
echo "   - Admin > Connections"
echo "   - Ajouter 'databricks_default' avec votre token"
echo ""
