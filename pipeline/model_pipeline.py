# ==============================================================================
# VERTEXAI TIME SERIES FORECASTING MLOPS PIPELINE
# ==============================================================================

"""
Time Series Forecasting Pipeline

This module defines the main Kubeflow pipeline for time series forecasting.
The pipeline orchestrates the entire ML workflow from data loading to model deployment.
"""

# pipeline/forecasting_pipeline.py
from kfp import dsl
from kfp.dsl import pipeline, Output, Artifact

# Import components
from components.check_bucket import check_and_create_gcs_bucket
from components.data_loading import load_and_create_dataset_component
from components.data_preprocessing import data_preprocessing_component
from components.model_trainer import model_training_component
from components.model_evaluation import model_evaluation_component
from components.model_deployment import model_deployment_component
from components.model_endpoint import model_endpoint_component


@pipeline(
    name="custom-time-series-forecasting-pipeline",
    description="MLOps pipeline for custom Time Series forecasting model training and deployment.",
)
def _forecasting_pipeline_(
    project_id: str,
    region: str,
    sql_query: str,
    target_variable: str,
    date_variable: str,
    bucket_name: str,
    dataset_display_name: str,
    serving_container_image_uri: str,
    model_display_name: str,
    endpoint_display_name: str,
    model_deployment_path: str,
    model_version: str,
):

    # --- 0. Check or Create Cloud Storage Bucket Task ---
    bucket_check_task = check_and_create_gcs_bucket(
        bucket_name=bucket_name,
        project_id=project_id,
        region=region,
    ).set_display_name("0. Check or Create Cloud Storage Bucket Task")

    # --- 1. Data Loading and Creating Dataset Task ---
    data_load_task = load_and_create_dataset_component(
        project_id=project_id,
        sql_query=sql_query,
        bucket_status=bucket_check_task.outputs["bucket_status"],
    ).set_display_name("1. Data Loading and Creating Dataset Task")

    # --- 2. Data Preprocessing for Model Training Task ---
    # NOTE: This component should now perform time-series feature engineering (lags, holidays, etc.)
    data_preprocess_task = data_preprocessing_component(
        project_id=project_id,
        region=region,
        date_variable=date_variable,
        target_variable=target_variable,
        dataset_display_name=dataset_display_name,
        training_data=data_load_task.outputs["training_data"],
    ).set_display_name("2. Preprocessing Dataset for Forecasting Task")

    # --- 3. Training of Model for Time Series Forecasting Task ---
    model_training_task = model_training_component(
        date_variable=date_variable,
        target_variable=target_variable,
        preprocess_data=data_preprocess_task.outputs["preprocess_data"],
    ).set_display_name("3. Training of Time Series Forecasting Models")

    # --- 4. Model Evaluation and Promotion Check (using RMSE) ---
    model_evaluation_task = model_evaluation_component(
        project_id=project_id,
        region=region,
        model_display_name=model_display_name,
        # preprocess_data=data_preprocess_task.outputs["preprocess_data"],
        new_model=model_training_task.outputs["model"],
    ).set_display_name("4. Model Evaluation and Promotion Check (RMSE)")

    # --- 5. Trained Model Deployment Task ---
    with dsl.If(model_evaluation_task.outputs["promotion_status"] == "True"):
        model_deployment_task = model_deployment_component(
            project_id=project_id,
            region=region,
            serving_container_image_uri=serving_container_image_uri,
            model_display_name=model_display_name,
            bucket_name=bucket_name,
            model_deployment_path=model_deployment_path,
            model_version=model_version,
            model=model_training_task.outputs["model"],
            feature_importance_artifact=model_training_task.outputs[
                "feature_importance_artifact"
            ],
        ).set_display_name("5. Trained Forecasting Model Deployment Task")

    # --- 6. Endpoint Management & Model Serving Task ---
    # NOTE: Keeping this disabled block for now, as Vertex AI serving statistical TS models
    # (Prophet, ARIMA) requires a custom container, which is often easier via Cloud Run/Streamlit.
    # If the model is a simple Sklearn/XGBoost, the built-in container might work.

    # To re-enable endpoint deployment, uncomment the code below:
    endpoint_management_task = model_endpoint_component(
        project_id=project_id,
        region=region,
        endpoint_display_name=endpoint_display_name,
        model_display_name=model_display_name,
        uploaded_model_resource_name=model_deployment_task.outputs[
            "uploaded_model_resource_name"
        ],
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=1,
        traffic_percentage=100,
    ).set_display_name("6. Endpoint Management & Model Serving Task")
    endpoint_management_task.after(model_deployment_task)
