# ==============================================================================
# ENDPOINT MANAGEMENT & MODEL SERVING TASK
# ==============================================================================

"""
Model Endpoint Component

This component deploys the trained model to Vertex AI Endpoints for serving.
"""

# components/endpoint_management.py
from kfp.dsl import component, Output, Metrics
from typing import NamedTuple


@component(
    packages_to_install=[
        "google-cloud-aiplatform",
        "google-api-core",
    ],
    base_image="python:3.10",
)
def model_endpoint_component(
    project_id: str,
    region: str,
    endpoint_display_name: str,
    model_display_name: str,
    uploaded_model_resource_name: str,
    machine_type: str,
    min_replica_count: int,
    max_replica_count: int,
    traffic_percentage: int,
    endpoint_details: Output[Metrics],
) -> NamedTuple("Outputs", [("endpoint_id", str), ("deployment_status", str)]):  # type: ignore
    """
    Manages Vertex AI Endpoint for *Time Series Forecasting* model deployment
    using the deploy-then-undeploy pattern for atomic replacement.
    """
    from google.cloud import aiplatform
    from google.cloud.aiplatform.models import Endpoint
    import logging
    from google.api_core import exceptions
    from collections import namedtuple

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.info(f"✅ {25*'#'} FORECASTING ENDPOINT MANAGEMENT TASK {25*'#'}")
    status_message = "Deployment failed due to unhandled exception."

    # Define the output NamedTuple structure for the return
    Outputs = namedtuple("Outputs", ["endpoint_id", "deployment_status"])

    try:
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=region)

        # --- 1. Check/Create Endpoint ---
        def get_or_create_endpoint() -> Endpoint:
            endpoints = aiplatform.Endpoint.list(
                filter=f'display_name="{endpoint_display_name}"'
            )
            if endpoints:
                endpoint = endpoints[0]
                logging.info(f"✅ Found existing endpoint: {endpoint.resource_name}")
                return endpoint
            else:
                logging.info(
                    f"⚠️ Endpoint '{endpoint_display_name}' not found. Creating new endpoint..."
                )
                endpoint = aiplatform.Endpoint.create(
                    display_name=endpoint_display_name,
                    project=project_id,
                    location=region,
                    description=f"Vertex AI Endpoint for Time Series Forecasting Model: {model_display_name}",  # Contextual update
                    sync=True,
                )
                # CRITICAL: Reload the endpoint object after creation to ensure it is fully populated.
                endpoint = aiplatform.Endpoint(endpoint.resource_name)
                logging.info(
                    f"✅ Created and loaded new endpoint: {endpoint.resource_name}"
                )
                return endpoint

        # --- 2. Load Model & Endpoint ---
        endpoint = get_or_create_endpoint()

        # Load the model resource directly using the resource name from the upload step
        new_model = aiplatform.Model(uploaded_model_resource_name)
        logging.info(f"✅ Loaded model resource: {new_model.resource_name}")

        # --- 3. Determine Deployed Model ID to be Undeployed ---
        deployed_model_id_to_undeploy = None
        current_deployed_models = endpoint.gca_resource.deployed_models

        if current_deployed_models:
            # If models are deployed, store the ID of the first (or only) deployed model for cleanup.
            deployed_model_id_to_undeploy = current_deployed_models[0].id
            logging.info(
                f"ℹ️ Existing deployed model found. ID to undeploy after deployment: {deployed_model_id_to_undeploy}"
            )

        # --- 4. Deploy New Model (Gets 100% traffic for atomic replacement) ---
        logging.info("🚀 Deploying new model...")

        # Deploy the new model and direct all traffic (traffic_percentage) to it.
        # Note: The key is '0' which references the model being deployed in this request.
        traffic_config = {"0": traffic_percentage}

        deployed_model = endpoint.deploy(
            model=new_model,
            deployed_model_display_name=model_display_name,
            machine_type=machine_type,
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count,
            traffic_split=traffic_config,
            sync=True,
            deploy_request_timeout=1800,
        )

        logging.info(
            "✅ New forecasting model deployed successfully and received traffic."
        )

        # --- 5. Clean up (Undeploy old model if it existed) ---
        if deployed_model_id_to_undeploy:
            logging.info(
                f"🗑️ Explicitly undeploying previous model ID: {deployed_model_id_to_undeploy}"
            )
            endpoint.undeploy(
                deployed_model_id=deployed_model_id_to_undeploy, sync=True
            )
            logging.info("✅ Previous model undeployed successfully.")

        status_message = "SUCCESS: Forecasting model deployed."

        # Log metrics
        endpoint_details.log_metric("deployment_status", 1)
        endpoint_details.log_metric("endpoint_resource_name", endpoint.resource_name)

        # --- 6. Return Outputs ---
        return Outputs(endpoint_id=endpoint.name, deployment_status=status_message)

    except exceptions.GoogleAPICallError as e:
        error_msg = f"❌ Deployment failed: {e}"
        logging.error(error_msg)
        status_message = f"FAILURE: {str(e)}"
        raise RuntimeError(error_msg)

    except Exception as e:
        error_msg = f"❌ Endpoint management failed: {e}"
        logging.error(error_msg)
        raise RuntimeError(error_msg)
