"""
Integration tests for Databricks batch pipeline
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from src.pipelines.batch_pipeline import DatabricksBatchPipeline, get_batch_pipeline


class TestDatabricksBatchPipeline:
    """Tests for DatabricksBatchPipeline"""

    @patch('src.cloud.databricks.WorkspaceClient')
    def test_pipeline_initialization(self, mock_workspace_client):
        """Test batch pipeline initialization"""
        pipeline = DatabricksBatchPipeline(
            databricks_host='https://test.cloud.databricks.com',
            databricks_token='test-token',
            cluster_id='test-cluster-id'
        )
        
        assert pipeline.cluster_id == 'test-cluster-id'
        assert pipeline.job_manager is not None
        assert pipeline.notebook_executor is not None

    @patch('src.cloud.databricks.WorkspaceClient')
    def test_pipeline_execute_success(self, mock_workspace_client):
        """Test successful pipeline execution"""
        mock_client = Mock()
        mock_run = Mock()
        mock_run.run_id = 99999
        mock_run.state.value = 'SUCCEEDED'
        mock_client.jobs.submit.return_value = mock_run
        mock_client.jobs.get_run.return_value = mock_run
        mock_workspace_client.return_value = mock_client
        
        pipeline = DatabricksBatchPipeline(
            databricks_host='https://test.cloud.databricks.com',
            databricks_token='test-token',
            cluster_id='test-cluster-id'
        )
        
        result = pipeline.execute(date_range_days=1)
        
        assert result['status'] == 'success'
        assert result['run_id'] == 99999

    @patch('src.cloud.databricks.WorkspaceClient')
    def test_pipeline_execute_failure(self, mock_workspace_client):
        """Test failed pipeline execution"""
        mock_client = Mock()
        mock_run = Mock()
        mock_run.run_id = 88888
        mock_run.state.value = 'FAILED'
        mock_client.jobs.submit.return_value = mock_run
        mock_client.jobs.get_run.return_value = mock_run
        mock_workspace_client.return_value = mock_client
        
        pipeline = DatabricksBatchPipeline(
            databricks_host='https://test.cloud.databricks.com',
            databricks_token='test-token',
            cluster_id='test-cluster-id'
        )
        
        result = pipeline.execute(date_range_days=1)
        
        assert result['status'] == 'failed'
        assert 'error' in result

    @patch.dict('os.environ', {
        'DATABRICKS_HOST': 'https://test.cloud.databricks.com',
        'DATABRICKS_TOKEN': 'test-token',
        'DATABRICKS_CLUSTER_ID': 'test-cluster'
    })
    @patch('src.cloud.databricks.WorkspaceClient')
    def test_factory_function_from_env(self, mock_workspace_client):
        """Test factory function creates pipeline from environment variables"""
        pipeline = get_batch_pipeline()
        
        assert pipeline is not None
        assert isinstance(pipeline, DatabricksBatchPipeline)
        assert pipeline.cluster_id == 'test-cluster'

    @patch.dict('os.environ', {})
    def test_factory_function_missing_env_vars(self):
        """Test factory function raises error when env vars missing"""
        with pytest.raises(ValueError, match="Missing required Databricks environment variables"):
            get_batch_pipeline()
