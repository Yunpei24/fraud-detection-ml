# Tests for Drift Detection Module

## Overview

This directory contains comprehensive unit and integration tests for the drift detection module.

## Test Structure

```
tests/
├── conftest.py              # Shared pytest fixtures
├── __init__.py
│
├── unit/                    # Unit tests (fast, isolated)
│   ├── test_data_drift.py
│   ├── test_target_drift.py
│   ├── test_concept_drift.py
│   ├── test_adwin.py
│   ├── test_alert_manager.py
│   ├── test_trigger.py
│   └── test_statistical_tests.py
│
└── integration/             # Integration tests (slower, require dependencies)
    ├── test_pipeline_hourly.py
    ├── test_pipeline_daily.py
    ├── test_database_integration.py
    └── test_end_to_end.py
```

## Installation

Install test dependencies:

```bash
pip install -r requirements-test.txt
```

Or install with main dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-test.txt
```

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Unit Tests Only

```bash
pytest tests/unit/ -v
```

### Run Integration Tests Only

```bash
pytest tests/integration/ -v
```

### Run Specific Test File

```bash
pytest tests/unit/test_data_drift.py -v
```

### Run Specific Test Class

```bash
pytest tests/unit/test_data_drift.py::TestDataDriftDetector -v
```

### Run Specific Test Method

```bash
pytest tests/unit/test_data_drift.py::TestDataDriftDetector::test_compute_psi_no_drift -v
```

### Run Tests by Marker

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only slow tests
pytest -m slow

# Run tests requiring database
pytest -m database

# Exclude slow tests
pytest -m "not slow"
```

## Test Coverage

### Generate Coverage Report

```bash
pytest --cov=src --cov-report=html --cov-report=term-missing
```

View HTML report:

```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Coverage Requirements

- **Target Coverage**: 80%+
- **Minimum Coverage**: 70%

## Test Options

### Verbose Output

```bash
pytest -v
```

### Show Print Statements

```bash
pytest -s
```

### Stop on First Failure

```bash
pytest -x
```

### Run Last Failed Tests

```bash
pytest --lf
```

### Run in Parallel

```bash
pytest -n auto  # Uses all CPU cores
pytest -n 4     # Uses 4 cores
```

## Test Markers

Available markers:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.database` - Tests requiring database

## Writing New Tests

### Unit Test Template

```python
import pytest
from src.module import MyClass

@pytest.mark.unit
class TestMyClass:
    """Test suite for MyClass."""
    
    def test_my_method(self, test_settings):
        """Test description."""
        obj = MyClass(test_settings)
        result = obj.my_method()
        assert result == expected_value
```

### Integration Test Template

```python
import pytest
from unittest.mock import patch

@pytest.mark.integration
class TestMyIntegration:
    """Integration test suite."""
    
    @patch('src.module.external_dependency')
    def test_integration(self, mock_dep, test_settings):
        """Test integration."""
        # Test code here
        assert condition
```

## Fixtures

Available fixtures (from `conftest.py`):

- `test_settings` - Test configuration
- `baseline_data` - Synthetic baseline dataset
- `current_data_no_drift` - Current data without drift
- `current_data_with_drift` - Current data with drift
- `predictions_and_labels` - Predictions for concept drift testing
- `drift_results_sample` - Sample drift detection results
- `temp_output_dir` - Temporary directory for test outputs

## Mocking

### Mock Database

```python
@patch('src.storage.database.Session')
def test_with_mock_db(mock_session):
    mock_session_instance = MagicMock()
    mock_session.return_value = mock_session_instance
    # Test code
```

### Mock External APIs

```python
@patch('src.module.requests.post')
def test_with_mock_api(mock_post):
    mock_post.return_value.status_code = 200
    # Test code
```

## Continuous Integration

Tests are automatically run on:

- **Pre-commit**: Fast unit tests
- **Pull Requests**: All tests
- **Merge to Main**: All tests + coverage report

### GitHub Actions Configuration

```yaml
- name: Run Tests
  run: |
    pytest tests/ -v --cov=src --cov-report=xml
    
- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

## Troubleshooting

### Import Errors

If you see import errors, ensure you're in the drift directory:

```bash
cd /path/to/drift
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest
```

### Database Connection Errors

Integration tests mock database connections by default. To test with real database:

```bash
# Start test database
docker-compose -f docker-compose.test.yml up -d

# Run database tests
pytest -m database
```

### Slow Tests

Skip slow tests during development:

```bash
pytest -m "not slow"
```

## Test Data

Test data is generated synthetically using:

- **NumPy**: Random distributions
- **Pandas**: DataFrames
- **Faker**: Realistic fake data (if needed)

No real fraud data is used in tests.

## Best Practices

1. **Isolation**: Each test should be independent
2. **Fast**: Unit tests should run in milliseconds
3. **Clear**: Test names should describe what they test
4. **Comprehensive**: Test both success and failure cases
5. **Mocking**: Mock external dependencies (DB, APIs)
6. **Assertions**: Use clear, specific assertions
7. **Coverage**: Aim for 80%+ code coverage

## Performance

Expected test execution times:

- **Unit Tests**: < 30 seconds
- **Integration Tests**: 1-3 minutes
- **All Tests**: 2-4 minutes
- **Parallel Execution**: 30-60 seconds

## Contact

For questions about tests, see:

- **Main README**: `../README.md`
- **Documentation**: `../docs/`
- **Contributing Guide**: `../CONTRIBUTING.md`
