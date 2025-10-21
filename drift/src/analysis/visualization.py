"""
Visualization module for drift detection.

This module provides functions to create plots and visualizations for:
- Drift timeline over time
- Feature distribution comparisons
- Confusion matrix evolution
- Drift reports
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
from pathlib import Path
import structlog

logger = structlog.get_logger(__name__)


def plot_drift_timeline(
    drift_data: pd.DataFrame,
    drift_type: str = "all",
    output_path: Optional[str] = None
) -> go.Figure:
    """
    Plot drift scores over time.
    
    Args:
        drift_data: DataFrame with columns [timestamp, drift_type, drift_score, threshold]
        drift_type: Type of drift to plot ("data", "target", "concept", or "all")
        output_path: Optional path to save the plot
        
    Returns:
        Plotly figure object
    """
    logger.info("plotting_drift_timeline", drift_type=drift_type)
    
    # Filter data if specific type requested
    if drift_type != "all":
        drift_data = drift_data[drift_data["drift_type"] == drift_type]
    
    # Create figure
    fig = go.Figure()
    
    # Plot drift scores by type
    for dt in drift_data["drift_type"].unique():
        data_subset = drift_data[drift_data["drift_type"] == dt]
        
        fig.add_trace(go.Scatter(
            x=data_subset["timestamp"],
            y=data_subset["drift_score"],
            mode="lines+markers",
            name=f"{dt.capitalize()} Drift",
            line=dict(width=2),
            marker=dict(size=6)
        ))
        
        # Add threshold line
        if "threshold" in data_subset.columns:
            threshold = data_subset["threshold"].iloc[0]
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"{dt.capitalize()} Threshold: {threshold}",
                annotation_position="right"
            )
    
    # Update layout
    fig.update_layout(
        title="Drift Detection Timeline",
        xaxis_title="Timestamp",
        yaxis_title="Drift Score",
        hovermode="x unified",
        template="plotly_white",
        height=600,
        width=1200
    )
    
    # Save if path provided
    if output_path:
        fig.write_html(output_path)
        logger.info("drift_timeline_saved", path=output_path)
    
    return fig


def plot_feature_distributions(
    current_data: pd.DataFrame,
    baseline_data: pd.DataFrame,
    features: List[str],
    output_path: Optional[str] = None
) -> go.Figure:
    """
    Plot feature distribution comparisons between current and baseline.
    
    Args:
        current_data: Current data DataFrame
        baseline_data: Baseline data DataFrame
        features: List of feature names to plot
        output_path: Optional path to save the plot
        
    Returns:
        Plotly figure object
    """
    logger.info("plotting_feature_distributions", num_features=len(features))
    
    # Calculate number of rows and columns for subplots
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # Create subplots
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=features,
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # Plot each feature
    for idx, feature in enumerate(features):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        # Baseline histogram
        fig.add_trace(
            go.Histogram(
                x=baseline_data[feature],
                name="Baseline",
                opacity=0.6,
                marker_color="blue",
                showlegend=(idx == 0)
            ),
            row=row,
            col=col
        )
        
        # Current histogram
        fig.add_trace(
            go.Histogram(
                x=current_data[feature],
                name="Current",
                opacity=0.6,
                marker_color="red",
                showlegend=(idx == 0)
            ),
            row=row,
            col=col
        )
    
    # Update layout
    fig.update_layout(
        title="Feature Distribution Comparison: Baseline vs Current",
        template="plotly_white",
        height=300 * n_rows,
        width=1400,
        barmode="overlay"
    )
    
    # Save if path provided
    if output_path:
        fig.write_html(output_path)
        logger.info("feature_distributions_saved", path=output_path)
    
    return fig


def plot_confusion_matrix_timeline(
    metrics_data: pd.DataFrame,
    output_path: Optional[str] = None
) -> go.Figure:
    """
    Plot confusion matrix metrics over time.
    
    Args:
        metrics_data: DataFrame with columns [timestamp, recall, precision, fpr, f1_score]
        output_path: Optional path to save the plot
        
    Returns:
        Plotly figure object
    """
    logger.info("plotting_confusion_matrix_timeline")
    
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=["Recall", "Precision", "False Positive Rate", "F1 Score"],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Recall
    fig.add_trace(
        go.Scatter(
            x=metrics_data["timestamp"],
            y=metrics_data["recall"],
            mode="lines+markers",
            name="Recall",
            line=dict(color="green", width=2)
        ),
        row=1,
        col=1
    )
    
    # Precision
    fig.add_trace(
        go.Scatter(
            x=metrics_data["timestamp"],
            y=metrics_data["precision"],
            mode="lines+markers",
            name="Precision",
            line=dict(color="blue", width=2)
        ),
        row=1,
        col=2
    )
    
    # FPR
    fig.add_trace(
        go.Scatter(
            x=metrics_data["timestamp"],
            y=metrics_data["fpr"],
            mode="lines+markers",
            name="FPR",
            line=dict(color="red", width=2)
        ),
        row=2,
        col=1
    )
    
    # F1 Score
    fig.add_trace(
        go.Scatter(
            x=metrics_data["timestamp"],
            y=metrics_data["f1_score"],
            mode="lines+markers",
            name="F1 Score",
            line=dict(color="purple", width=2)
        ),
        row=2,
        col=2
    )
    
    # Update layout
    fig.update_layout(
        title="Model Performance Metrics Over Time",
        template="plotly_white",
        height=800,
        width=1400,
        showlegend=False
    )
    
    # Save if path provided
    if output_path:
        fig.write_html(output_path)
        logger.info("confusion_matrix_timeline_saved", path=output_path)
    
    return fig


def save_drift_report(
    drift_timeline: go.Figure,
    feature_distributions: go.Figure,
    confusion_matrix: go.Figure,
    output_dir: str,
    report_name: str = "drift_report"
) -> Dict[str, str]:
    """
    Save all drift visualizations to a directory.
    
    Args:
        drift_timeline: Drift timeline figure
        feature_distributions: Feature distributions figure
        confusion_matrix: Confusion matrix timeline figure
        output_dir: Directory to save reports
        report_name: Base name for report files
        
    Returns:
        Dictionary with paths to saved files
    """
    logger.info("saving_drift_report", output_dir=output_dir, report_name=report_name)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save individual plots
    paths = {}
    
    timeline_path = output_path / f"{report_name}_timeline.html"
    drift_timeline.write_html(str(timeline_path))
    paths["timeline"] = str(timeline_path)
    
    distributions_path = output_path / f"{report_name}_distributions.html"
    feature_distributions.write_html(str(distributions_path))
    paths["distributions"] = str(distributions_path)
    
    confusion_path = output_path / f"{report_name}_confusion.html"
    confusion_matrix.write_html(str(confusion_path))
    paths["confusion"] = str(confusion_path)
    
    logger.info("drift_report_saved", paths=paths)
    
    return paths


def plot_psi_heatmap(
    psi_scores: Dict[str, float],
    threshold: float = 0.25,
    output_path: Optional[str] = None
) -> go.Figure:
    """
    Create a heatmap of PSI scores for all features.
    
    Args:
        psi_scores: Dictionary of feature names to PSI scores
        threshold: PSI threshold for highlighting
        output_path: Optional path to save the plot
        
    Returns:
        Plotly figure object
    """
    logger.info("plotting_psi_heatmap", num_features=len(psi_scores))
    
    # Prepare data
    features = list(psi_scores.keys())
    scores = list(psi_scores.values())
    
    # Create color scale (green < 0.1, yellow 0.1-0.25, red > 0.25)
    colors = []
    for score in scores:
        if score < 0.1:
            colors.append("green")
        elif score < threshold:
            colors.append("yellow")
        else:
            colors.append("red")
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=features,
            y=scores,
            marker=dict(
                color=scores,
                colorscale="RdYlGn_r",
                showscale=True,
                colorbar=dict(title="PSI Score")
            ),
            text=[f"{score:.3f}" for score in scores],
            textposition="outside"
        )
    ])
    
    # Add threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold: {threshold}"
    )
    
    # Update layout
    fig.update_layout(
        title="Population Stability Index (PSI) by Feature",
        xaxis_title="Features",
        yaxis_title="PSI Score",
        template="plotly_white",
        height=600,
        width=1200,
        showlegend=False
    )
    
    # Save if path provided
    if output_path:
        fig.write_html(output_path)
        logger.info("psi_heatmap_saved", path=output_path)
    
    return fig
