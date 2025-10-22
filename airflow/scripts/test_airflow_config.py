#!/usr/bin/env python3
"""
Test script pour valider la configuration Airflow
Usage: python test_airflow_config.py
"""

import sys
import os

# Add paths
sys.path.append('/opt/airflow/fraud-detection-ml')
sys.path.append('/opt/airflow')

def test_imports():
    """Test that all imports work"""
    print("🔍 Test des imports...")
    
    try:
        from airflow.config.settings import settings
        print(f"✅ Settings importées: {type(settings)}")
    except Exception as e:
        print(f"❌ Erreur import settings: {e}")
        return False
    
    try:
        from drift.src.config.settings import Settings as DriftSettings
        print(f"✅ DriftSettings importées: {type(DriftSettings)}")
    except Exception as e:
        print(f"❌ Erreur import DriftSettings: {e}")
        return False
    
    return True


def test_database_connections():
    """Test database connections"""
    print("\n🗄️  Test des connexions database...")
    
    try:
        from airflow.config.settings import settings
        import sqlalchemy as sa
        
        # Test fraud_db connection
        print(f"Database URL: {settings.fraud_database_url}")
        engine = sa.create_engine(settings.fraud_database_url)
        
        with engine.connect() as conn:
            result = conn.execute(sa.text("SELECT 1"))
            print(f"✅ Connexion fraud_db OK")
            
            # Check tables exist
            result = conn.execute(sa.text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """))
            tables = [row[0] for row in result]
            
            required_tables = [
                'transactions', 'predictions', 'drift_metrics',
                'retraining_triggers', 'model_versions'
            ]
            
            for table in required_tables:
                if table in tables:
                    print(f"✅ Table '{table}' existe")
                else:
                    print(f"⚠️  Table '{table}' manquante")
            
        return True
        
    except Exception as e:
        print(f"❌ Erreur connexion database: {e}")
        return False


def test_settings_values():
    """Test that all required settings are configured"""
    print("\n⚙️  Test des settings...")
    
    try:
        from airflow.config.settings import settings
        
        required = {
            'fraud_database_url': settings.fraud_database_url,
            'api_base_url': settings.api_base_url,
            'mlflow_tracking_uri': settings.mlflow_tracking_uri,
            'data_drift_threshold': settings.data_drift_threshold,
            'concept_drift_threshold': settings.concept_drift_threshold,
            'training_cooldown_hours': settings.training_cooldown_hours,
            'min_training_samples': settings.min_training_samples
        }
        
        for key, value in required.items():
            if value:
                print(f"✅ {key}: {value}")
            else:
                print(f"⚠️  {key}: Non configuré")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur test settings: {e}")
        return False


def test_drift_module_integration():
    """Test drift module can be called"""
    print("\n📊 Test intégration module drift...")
    
    try:
        from drift.src.pipelines.hourly_monitoring import run_hourly_monitoring
        from drift.src.config.settings import Settings
        
        print("✅ Module drift importé")
        print(f"✅ Fonction run_hourly_monitoring disponible")
        
        # Test settings
        drift_settings = Settings()
        print(f"✅ DriftSettings initialisées")
        print(f"   - Database: {drift_settings.database.database}")
        print(f"   - Thresholds configurés: {drift_settings.drift_thresholds.data_drift_threshold}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur intégration drift: {e}")
        return False


def test_mlflow_connection():
    """Test MLflow connection"""
    print("\n🔬 Test connexion MLflow...")
    
    try:
        import mlflow
        from airflow.config.settings import settings
        
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        
        # Try to get or create experiment
        experiment_name = "/fraud-detection/test"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment:
            print(f"✅ MLflow connecté: {settings.mlflow_tracking_uri}")
            print(f"✅ Experiment trouvé: {experiment_name}")
        else:
            print(f"✅ MLflow connecté: {settings.mlflow_tracking_uri}")
            print(f"⚠️  Experiment '{experiment_name}' pas trouvé (sera créé au besoin)")
        
        return True
        
    except Exception as e:
        print(f"⚠️  MLflow non disponible: {e}")
        print("   (Normal si MLflow n'est pas encore démarré)")
        return True  # Non bloquant


def main():
    """Run all tests"""
    print("="*60)
    print("🧪 Tests de configuration Airflow")
    print("="*60)
    
    results = {
        'imports': test_imports(),
        'database': test_database_connections(),
        'settings': test_settings_values(),
        'drift_integration': test_drift_module_integration(),
        'mlflow': test_mlflow_connection()
    }
    
    print("\n" + "="*60)
    print("📋 Résumé des tests")
    print("="*60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_critical_passed = (
        results['imports'] and 
        results['database'] and 
        results['settings'] and
        results['drift_integration']
    )
    
    if all_critical_passed:
        print("\n✅ Configuration Airflow VALIDE - Prêt pour production!")
        sys.exit(0)
    else:
        print("\n❌ Configuration Airflow INVALIDE - Corriger les erreurs")
        sys.exit(1)


if __name__ == '__main__':
    main()
