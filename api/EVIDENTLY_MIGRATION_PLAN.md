# Evidently AI Migration Plan

## Current State
- Complex import logic in `drift_service.py` with try/except blocks
- Dependencies on separate `fraud_detection_drift` module
- Error-prone fallback mechanisms
- Hard to maintain and debug

## Target State
- Clean Evidently AI integration in `evidently_drift_service.py`
- No complex import logic
- Direct dependency on `evidently` package
- Comprehensive drift detection capabilities

## Migration Steps

### Phase 1: Preparation ✅ COMPLETE
1. ✅ Add `evidently==0.4.16` to `api/requirements.txt`
2. ✅ Create `evidently_drift_service.py` with comprehensive drift detection
3. ✅ Update drift routes to use new service (when ready)

### Phase 2: Database Schema Updates
1. Create drift_results table:
```sql
CREATE TABLE IF NOT EXISTS drift_analysis_results (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    analysis_window VARCHAR(20),
    reference_window VARCHAR(20),
    data_drift JSONB,
    target_drift JSONB,
    concept_drift JSONB,
    multivariate_drift JSONB,
    drift_summary JSONB
);

CREATE TABLE IF NOT EXISTS drift_reports (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    summary JSONB,
    recommendations TEXT[],
    alerts JSONB,
    severity VARCHAR(20)
);
```

2. Add indexes for performance:
```sql
CREATE INDEX idx_drift_analysis_timestamp ON drift_analysis_results(timestamp);
CREATE INDEX idx_drift_reports_timestamp ON drift_reports(timestamp);
CREATE INDEX idx_drift_reports_severity ON drift_reports(severity);
```

### Phase 3: Service Migration
1. Update `drift_service.py` imports to use `evidently_drift_service.py`
2. Replace complex import logic with simple Evidently AI imports
3. Update all drift-related routes to use new service
4. Remove dependency on separate drift module

### Phase 4: Testing & Validation
1. Test drift detection endpoints with Evidently AI
2. Validate drift results are stored correctly
3. Ensure backward compatibility during transition
4. Update integration tests

### Phase 5: Cleanup
1. Remove legacy `drift_service.py` complex import logic
2. Remove references to separate drift module
3. Update documentation
4. Remove unused drift module dependencies

## Advanced Drift Detection Capabilities ✅ IMPLEMENTED

### 1. All Three Drift Types
- **Data Drift**: Feature distribution changes using `DatasetDriftMetric` and `ColumnDriftMetric`
- **Target Drift**: Fraud rate changes using PSI (Population Stability Index)
- **Concept Drift**: Feature-target relationship changes using correlation analysis

### 2. Statistical Tests
- **KS Test**: `ks_stattest` for continuous features
- **Chi-squared Test**: `chisquare_stattest` for categorical features
- **PSI Test**: `psi_stat_test` for target drift detection
- **Multiple other tests**: Anderson-Darling, Jensen-Shannon, Wasserstein, etc.

### 3. Sliding Windows for Continuous Monitoring
- `run_sliding_window_analysis()` method for time-based drift monitoring
- Configurable window sizes and step intervals
- Historical drift pattern analysis

### 4. Multivariate Drift Detection
- `TestAllFeaturesValueDrift` for comprehensive feature analysis
- `TestShareOfDriftedColumns` and `TestNumberOfDriftedColumns`
- Dataset-level drift assessment

### 5. Automated Drift Reports
- `generate_drift_report()` with severity assessment
- Actionable recommendations based on drift types detected
- Alert system with configurable thresholds

## Benefits of Migration
- **Simplified Architecture**: No complex import logic
- **Better Maintainability**: Clean, readable code
- **Enhanced Capabilities**: Full Evidently AI feature set
- **Improved Reliability**: No fallback mechanisms needed
- **Better Performance**: Direct integration without module loading overhead

## Rollback Plan
If issues arise during migration:
1. Keep old `drift_service.py` as backup
2. Switch back to legacy service by updating imports
3. Investigate issues with Evidently AI integration
4. Gradually migrate once issues are resolved

## Timeline
- Phase 1 (Preparation): ✅ Complete
- Phase 2 (Database): 1-2 days
- Phase 3 (Migration): 2-3 days
- Phase 4 (Testing): 1-2 days
- Phase 5 (Cleanup): 1 day

Total estimated time: 1 week