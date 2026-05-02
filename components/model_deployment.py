# ==============================================================================
# FORECASTING MODEL DEPLOYMENT TASK
# ==============================================================================

"""
Model Deployment Component

This component deploys the trained model to Vertex AI Endpoints for serving.
"""

# components/deployment_forecasting_only.py
from kfp.dsl import component, Input, Output, Model, Artifact, Metrics
from typing import NamedTuple


@component(
    packages_to_install=[
        "google-cloud-aiplatform",
        "joblib",
        "google-cloud-storage",
        "google-api-core",
    ],
    base_image="python:3.10",
)
def model_deployment_component(
    project_id: str,
    region: str,
    serving_container_image_uri: str,
    model_display_name: str,
    bucket_name: str,
    model_deployment_path: str,
    model_version: str,
    model: Input[Model],
    feature_importance_artifact: Input[Artifact],
    deployment_status: Output[Metrics],
) -> NamedTuple("Outputs", [("uploaded_model_resource_name", str)]):  # type: ignore
    """
    Uploads the trained time series model to Vertex AI Model Registry and pushes
    artifacts (model, feature importance placeholder, metadata) to GCS for reference.
    """
    import os
    import json
    from google.cloud import storage
    from datetime import datetime
    from google.cloud import aiplatform
    import logging
    from google.api_core import exceptions
    from collections import namedtuple

    # ------------------ Setup ------------------
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.info(f"✅ {25*'#'} FORECASTING MODEL DEPLOYMENT TASK {25*'#'}")

    # Define the output NamedTuple structure for the return
    Outputs = namedtuple("Outputs", ["uploaded_model_resource_name"])

    try:
        # 1. Initialize Vertex AI client
        aiplatform.init(project=project_id, location=region)

        # 2. Upload the Model to Vertex AI Model Registry
        logging.info("⏳ Starting Model Upload to Vertex AI Model Registry...")
        model_artifact_uri = model.uri.strip("/")

        uploaded_model = aiplatform.Model.upload(
            display_name=model_display_name,
            artifact_uri=model_artifact_uri,
            serving_container_image_uri=serving_container_image_uri,
            description="Vertex AI MLOps Pipeline for Time Series Forecasting.",
            # The metrics saved in model.metadata during training are automatically carried over!
        )
        uploaded_model_resource_name = uploaded_model.resource_name

        logging.info(f"✅ Model uploaded to Vertex AI Model Registry.")
        logging.info(f"✅ Resource Name: {uploaded_model.resource_name}")

    except exceptions.GoogleAPICallError as e:
        logging.error(f"❌ Model upload failed: {e}")
        raise RuntimeError(f"❌ Model upload failed: {e}")

    try:
        # 3. Upload artifacts to GCS for reference
        client = storage.Client(project=project_id)
        bucket = client.bucket(bucket_name)
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )  # Changed format for cleaner paths
        logging.info(f"✅ GCS bucket '{bucket_name}' accessed.")

        # 3a. Upload the trained model (model.pkl)
        model_pipeline_blob_path = f"{model_deployment_path}/model.pkl"
        model_pipeline_blob = bucket.blob(model_pipeline_blob_path)
        model_file_path = os.path.join(model.path, "model.pkl")

        if os.path.exists(model_file_path):
            model_pipeline_blob.upload_from_filename(model_file_path)
            logging.info(
                f"✅ ✓ Model uploaded to: gs://{bucket_name}/{model_pipeline_blob_path}"
            )

        # 3b. Upload feature importance (feature_importance.json) - Placeholder for statistical models
        feature_importance_blob_path = (
            f"{model_deployment_path}/feature_importance.json"
        )
        feature_importance_blob = bucket.blob(feature_importance_blob_path)
        feature_importance_file_path = os.path.join(
            feature_importance_artifact.path, "feature_importance.json"
        )

        if os.path.exists(feature_importance_file_path):
            feature_importance_blob.upload_from_filename(feature_importance_file_path)
            logging.info(
                f"✅ ✓ Feature importance uploaded (or placeholder) to: gs://{bucket_name}/{feature_importance_blob_path}"
            )

        # 3c. Create and upload model metadata (Aligning with new RMSE metric)
        model_metadata = {
            "model_type": "TimeSeriesForecasting",
            "deployment_timestamp": f"forecasting_{model_version}_{timestamp}",
            "metrics": {
                # PULLING THE NEW METRICS FROM ARTIFACT METADATA
                "best_algorithm": getattr(model, "metadata", {}).get(
                    "algorithm", "N/A"
                ),
                "rmse_test": getattr(model, "metadata", {}).get("rmse_test", "N/A"),
                "mae_test": getattr(model, "metadata", {}).get("mae_test", "N/A"),
                "mape_test": getattr(model, "metadata", {}).get("mape_test", "N/A"),
            },
            "last_train_date": getattr(model, "metadata", {}).get(
                "last_train_date", "N/A"
            ),
            "best_params": getattr(model, "metadata", {}).get("best_params", "N/A"),
            "deployment_status": "deployed",
        }
        # --- Ensure Directory Exists ---
        metadata_blob_path = f"{model_deployment_path}/model_metadata.json"
        metadata_blob = bucket.blob(metadata_blob_path)
        metadata_blob.upload_from_string(json.dumps(model_metadata, indent=4))
        logging.info(
            f"✅ ✓ Model metadata uploaded to: gs://{bucket_name}/{metadata_blob_path}"
        )

        # 3d. (Removed correlation data upload - unnecessary complexity for deployment)

        logging.info(f"✅ \n🎉 Model deployment artifacts published successfully!")
        deployment_status.log_metric("deployment_status", "deployed")

    except Exception as e:
        logging.error(f"❌ GCS/Artifact publishing failed: {e}")
        # Note: The model is already uploaded to the registry (step 2), but GCS artifacts failed.
        # We will still return the model resource name, but log the error.
        deployment_status.log_metric("deployment_status", "artifact_failure")

    # 4. Correctly return the NamedTuple output
    return Outputs(uploaded_model_resource_name=uploaded_model_resource_name)
