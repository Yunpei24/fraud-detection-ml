.PHONY: help setup start stop restart logs test clean dags

COMPOSE_FILE := ../../docker-compose.yml

help:
	@echo "Fraud Detection - Airflow Commands"
	@echo "=================================="
	@echo "setup     - Initial setup (use root setup-complete.sh instead)"
	@echo "start     - Start Airflow services"
	@echo "stop      - Stop Airflow services"
	@echo "restart   - Restart all services"
	@echo "logs      - Show scheduler logs"
	@echo "test      - Test configuration"
	@echo "dags      - List all DAGs"
	@echo "clean     - Clean all containers and volumes"
	@echo ""
	@echo "Access Airflow UI: http://localhost:8080"
	@echo "Username: airflow | Password: airflow"

setup:
	@echo "🚀 Pour le setup complet, utilisez:"
	@echo "   cd ../.. && bash setup-complete.sh"

start:
	@echo "▶️  Starting Airflow services..."
	@cd ../.. && docker-compose up -d airflow-webserver airflow-scheduler airflow-triggerer
	@echo "✅ Services started"
	@echo "Access UI: http://localhost:8080"

stop:
	@echo "⏹️  Stopping Airflow services..."
	@cd ../.. && docker-compose stop airflow-webserver airflow-scheduler airflow-triggerer
	@echo "✅ Services stopped"

restart: stop start

logs:
	@cd ../.. && docker-compose logs -f airflow-scheduler

logs-web:
	@cd ../.. && docker-compose logs -f airflow-webserver

test:
	@echo "🧪 Testing configuration..."
	@docker exec -it airflow-scheduler python /opt/airflow/scripts/test_airflow_config.py

dags:
	@echo "📊 Available DAGs:"
	@docker exec -it airflow-scheduler airflow dags list

dags-test:
	@echo "🧪 Testing DAG 02 (Drift Monitoring)..."
	@docker exec -it airflow-scheduler airflow dags test 02_drift_monitoring $$(date +%Y-%m-%d)

trigger-drift:
	@echo "▶️  Triggering Drift Monitoring DAG..."
	@docker exec -it airflow-scheduler airflow dags trigger 02_drift_monitoring

trigger-training:
	@echo "▶️  Triggering Training Pipeline DAG..."
	@docker exec -it airflow-scheduler airflow dags trigger 01_training_pipeline

trigger-feedback:
	@echo "▶️  Triggering Feedback Collection DAG..."
	@docker exec -it airflow-scheduler airflow dags trigger 03_feedback_collection

trigger-quality:
	@echo "▶️  Triggering Data Quality DAG..."
	@docker exec -it airflow-scheduler airflow dags trigger 04_data_quality

ps:
	@cd ../.. && docker-compose ps

db-check:
	@echo "🗄️  Checking fraud_db tables..."
	@docker exec -it postgres-fraud psql -U postgres -d fraud_db -c "\dt"

db-drift-metrics:
	@echo "📊 Latest drift metrics:"
	@docker exec -it postgres-fraud psql -U postgres -d fraud_db -c "SELECT * FROM drift_metrics ORDER BY detected_at DESC LIMIT 10;"

db-retraining:
	@echo "🔄 Retraining triggers:"
	@docker exec -it postgres-fraud psql -U postgres -d fraud_db -c "SELECT * FROM retraining_triggers ORDER BY triggered_at DESC LIMIT 5;"

clean:
	@echo "🧹 Cleaning Airflow..."
	@cd ../.. && docker-compose down airflow-webserver airflow-scheduler airflow-triggerer airflow-init
	@rm -rf logs/*
	@echo "✅ Cleaned"

clean-all:
	@echo "🧹 WARNING: This will remove ALL containers and volumes"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		cd ../.. && docker-compose down -v; \
		echo "✅ All cleaned"; \
	fi

shell-scheduler:
	@docker exec -it airflow-scheduler bash

shell-webserver:
	@docker exec -it airflow-webserver bash

shell-postgres:
	@docker exec -it postgres-fraud psql -U postgres -d fraud_db
