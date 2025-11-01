# Fraud Detection System Monitoring Setup

This directory contains comprehensive monitoring configuration for the fraud detection system, including Grafana dashboards, Prometheus alert rules, and notification configurations.

## Overview

The monitoring stack consists of:
- **4 Grafana Dashboards**: System overview, API performance, data pipeline, and drift detection
- **Prometheus Alert Rules**: 20+ alerts covering service health, performance, and business metrics
- **Alert Notifications**: Email, Slack, and webhook integrations
- **Cross-subscription Support**: Designed for multi-VM deployments

## Directory Structure

```
monitoring/
├── grafana/
│   ├── dashboards/
│   │   ├── fraud-detection-api-dashboard.json
│   │   ├── fraud-detection-data-pipeline-dashboard.json
│   │   ├── fraud-detection-drift-monitoring-dashboard.json
│   │   └── fraud-detection-system-overview-dashboard.json
│   └── provisioning/
│       └── notifiers/
│           └── alert-notifications.yml
└── prometheus/
    └── alert_rules.yml
```

## Dashboards

### 1. System Overview Dashboard (`fraud-detection-system-overview-dashboard.json`)
- **Service Health Status**: Real-time status of all services
- **Service Uptime**: How long each service has been running
- **Health Check Panels**: Individual service health indicators
- **System Throughput**: Combined request rates across all services
- **Error Rates**: Error rates across all services with thresholds
- **Resource Usage**: Memory and CPU usage for all services
- **Alert Summary**: Active alerts table
- **Service Dependencies**: Visual dependency graph

### 2. API Performance Dashboard (`fraud-detection-api-dashboard.json`)
- **Request Metrics**: Request rates, success rates, error rates
- **Latency Analysis**: Response time histograms and percentiles
- **Prediction Metrics**: Fraud prediction rates and accuracy
- **Cache Performance**: Cache hit rates and Redis metrics
- **Rate Limiting**: Rate limit hits and enforcement
- **System Resources**: CPU, memory, and connection pools
- **Error Analysis**: Error types and HTTP status codes

### 3. Data Pipeline Dashboard (`fraud-detection-data-pipeline-dashboard.json`)
- **Pipeline Health**: Success/failure rates for data processing
- **Data Volume**: Transaction volumes and throughput
- **Data Quality**: Validation errors and data issues
- **Processing Performance**: Step-by-step processing times
- **Queue Management**: Queue sizes and processing backlogs
- **Storage Metrics**: Database and file system usage
- **Alert Integration**: Pipeline failure alerts

### 4. Drift Detection Dashboard (`fraud-detection-drift-monitoring-dashboard.json`)
- **Drift Scores**: Model and feature drift measurements
- **Model Performance**: Accuracy, precision, recall over time
- **Drift Alerts**: Active drift detection alerts
- **Retraining Triggers**: When models need retraining
- **Feature Analysis**: Individual feature drift tracking
- **Historical Trends**: Drift patterns over time
- **Performance Impact**: How drift affects prediction accuracy

## Alert Rules

The system includes 20+ comprehensive alerts organized by category:

### Service Health Alerts
- Service down detection for all critical services
- Uptime monitoring with appropriate thresholds

### API Performance Alerts
- High error rates (>5%)
- High latency (95th percentile >2s)
- Rate limiting violations

### Data Pipeline Alerts
- Pipeline failure rates (>10%)
- Data quality validation errors
- Processing backlog warnings

### Drift Detection Alerts
- Model drift detection (score >0.7)
- Feature drift detection (score >0.8)
- Drift monitoring failures

### Model Performance Alerts
- Accuracy drops below 85%
- High false positive rates (>15%)

### Infrastructure Alerts
- High memory usage (<10% available)
- High CPU usage (>80%)
- Low disk space (<10% available)

### Business Logic Alerts
- Low prediction volume
- Unusual traffic patterns
- Training pipeline issues

## Deployment Instructions

### Prerequisites
- Docker and Docker Compose
- Prometheus running and configured to scrape your services
- Grafana instance with provisioning enabled

### 1. Configure Prometheus

Add the alert rules to your Prometheus configuration:

```yaml
# prometheus.yml
rule_files:
  - 'alert_rules.yml'

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### 2. Configure Grafana

1. **Enable Provisioning**: Ensure your Grafana instance has provisioning enabled
2. **Copy Dashboard Files**: Place the JSON dashboard files in your Grafana provisioning directory
3. **Configure Notifications**: Update the alert-notifications.yml with your contact details
4. **Set Data Source**: Ensure Prometheus is configured as a data source named "Prometheus"

### 3. Update Contact Information

Edit `monitoring/grafana/provisioning/notifiers/alert-notifications.yml`:

```yaml
# Update email addresses
addresses: your-alerts@company.com

# Update Slack webhook URL
url: https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
```

### 4. Environment Variables

Set these environment variables for cross-subscription monitoring:

```bash
# VM1 (Services)
export PROMETHEUS_PUBLIC_IP=vm1-public-ip
export GRAFANA_PUBLIC_IP=vm2-public-ip

# VM2 (Monitoring)
export PROMETHEUS_SCRAPE_TARGETS=vm1-public-ip:9090
```

## Metrics Requirements

Your services must expose these Prometheus metrics:

### API Service Metrics
- `fraud_api_requests_total`
- `fraud_api_errors_total`
- `fraud_api_request_duration_seconds`
- `fraud_predictions_total`
- `fraud_cache_hits_total`
- `fraud_rate_limit_exceeded_total`

### Data Pipeline Metrics
- `data_pipeline_transactions_processed_total`
- `data_pipeline_transactions_failed_total`
- `data_pipeline_validation_errors_total`
- `data_pipeline_queue_size`
- `data_pipeline_step_duration_seconds`

### Drift Detection Metrics
- `fraud_drift_score`
- `fraud_feature_drift_score`
- `fraud_model_accuracy`
- `fraud_false_positives_total`
- `fraud_drift_last_check_seconds`

### Infrastructure Metrics
- Standard Prometheus process metrics
- Node exporter metrics (optional)

## Alert Severity Levels

- **Critical**: Service down, data loss, security issues
- **Warning**: Performance degradation, high error rates
- **Info**: Informational alerts, maintenance reminders

## Troubleshooting

### Dashboards Not Loading
1. Check Grafana logs for JSON syntax errors
2. Verify Prometheus data source is correctly configured
3. Ensure metric names match exactly

### Alerts Not Firing
1. Check Prometheus targets are healthy
2. Verify metric names in alert rules
3. Check alert rule syntax with `promtool check rules`

### Notifications Not Working
1. Test webhook/Slack URLs manually
2. Check Grafana notification policy configuration
3. Verify email server settings

## Customization

### Adding New Metrics
1. Update service instrumentation
2. Add metrics to appropriate dashboard
3. Create alert rules if needed
4. Update this README

### Modifying Thresholds
Edit alert rules in `alert_rules.yml` and adjust threshold values based on your environment.

### Adding New Services
1. Create service-specific dashboard
2. Add service health alerts
3. Update system overview dashboard
4. Configure Prometheus scraping

## Support

For issues with monitoring setup:
1. Check service logs
2. Verify metric exposure
3. Test Prometheus queries manually
4. Review Grafana dashboard JSON syntax