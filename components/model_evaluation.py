# ==============================================================================
# TIME SERIES MODEL EVALUATION TASK
# ==============================================================================

"""
Model Evaluation Component

This component evaluates the performance of the trained model against the currently deployed model.
"""

# components/model_evaluation.py
from kfp.dsl import component, Input, Output, Model, Metrics
from typing import NamedTuple, Optional


@component(
    packages_to_install=[
        "numpy",
        "pandas",
        "joblib",
        "scikit-learn",
        "google-cloud-aiplatform",
        "google-api-core",
    ],
    base_image="python:3.10",
)
def model_evaluation_component(
    project_id: str,
    region: str,
    model_display_name: str,
    new_model: Input[Model],
    evaluation_metrics: Output[Metrics],
    target_metric_key: str = "rmse_test",
    promotion_threshold: float = -0.0001,
    min_acceptable_rmse: Optional[float] = None,
) -> NamedTuple("Outputs", [("promotion_status", str)]):  # type: ignore
    """
    Compares the newly trained time series model against the deployed model
    in the Vertex AI Model Registry based on the 'rmse_test' metric.

    Promotes if:
    1. It's the first deployment (and performance is acceptable).
    2. The new model's RMSE is STRICTLY LOWER (better) than the deployed model's RMSE.
    """
    from collections import namedtuple
    from google.cloud import aiplatform
    import logging
    import traceback
    from google.api_core import exceptions

    # ------------------ Setup ------------------
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info(f"✅ {25*'#'} TIME SERIES MODEL EVALUATION TASK {25*'#'}")

    outputs = namedtuple("Outputs", ["promotion_status"])
    default_failure_output = outputs(
        "False"
    )  # Default to not promoting on any critical error

    try:
        # Initialize Vertex AI
        logging.info(
            f"✅ Initializing Vertex AI: Project='{project_id}', Region='{region}'"
        )
        aiplatform.init(project=project_id, location=region)

        # --- 1. Get New Model Performance (from Artifact Metadata) ---
        new_metric_value = -1.0

        # NOTE: The Training component must save the final RMSE/R2/etc. to the model.metadata
        if target_metric_key in new_model.metadata:
            new_metric_value = float(new_model.metadata[target_metric_key])
            logging.info(
                f"✅ New Model Performance ({target_metric_key}): {new_metric_value:.4f}"
            )
        else:
            logging.error(
                f"❌ New model metadata missing key: '{target_metric_key}'. Cannot evaluate."
            )
            # Cannot proceed without the new model's metric
            return default_failure_output

        # --- 2. Get Deployed Model Performance (from Vertex AI Model Registry) ---
        deployed_metric_value = 10000000
        deployed_model_exists = False

        try:
            # Filter for the latest deployed model version
            filter_model_pipeline = f'display_name="{model_display_name}"'
            models = aiplatform.Model.list(
                filter=filter_model_pipeline, order_by="create_time desc"
            )

            if models:
                deployed_model_exists = True
                deployed_model = models[0]
                logging.info(
                    f"✅ Latest deployed model: {deployed_model.resource_name}"
                )

                # FAST METHOD: Try to get metric from model metadata
                if (
                    hasattr(deployed_model, "metadata")
                    and deployed_model.metadata
                    and target_metric_key in deployed_model.metadata
                ):
                    deployed_metric_value = float(
                        deployed_model.metadata[target_metric_key]
                    )
                    logging.info(
                        f"✅ Deployed Model {target_metric_key} (from metadata): {deployed_metric_value:.4f}"
                    )
                else:
                    logging.warning(
                        f"⚠️ Deployed model metadata missing key: '{target_metric_key}'. Assuming worst performance (inf)."
                    )
                    # Keep deployed_metric_value as inf if the metric is missing,
                    # forcing promotion (safest for MLOps pipeline on first run/missing data)
            else:
                logging.info(
                    "ℹ️ No deployed model found in registry. First deployment scenario."
                )

        except exceptions.GoogleAPICallError as e:
            logging.error(f"❌ Error querying deployed model registry: {e}")
            deployed_metric_value = (
                10000000  # Default to a very large number on error to allow promotion
            )

        # --- 3. Promotion Decision (Lower RMSE is Better) ---

        metric_difference = (
            new_metric_value - deployed_metric_value
        )  # Negative value means IMPROVEMENT

        # Check if a minimum acceptable error is defined and violated
        if min_acceptable_rmse is not None and new_metric_value > min_acceptable_rmse:
            reason = f"New model {target_metric_key} too high ({new_metric_value:.4f} > {min_acceptable_rmse:.4f})."
            should_promote = False

        # Promotion Logic
        elif (
            metric_difference < promotion_threshold
        ):  # Check for a strict improvement (negative difference)
            should_promote = True
            reason = f"New model {target_metric_key} improved by {-metric_difference:+.4f} (Strict Improvement Required)."

        elif not deployed_model_exists or deployed_metric_value == 10000000:
            should_promote = True
            reason = "No valid deployed model found. Promoting new model (First Deployment/Missing Deployed Metric)."

        else:
            should_promote = False
            reason = f"New model {target_metric_key} did not improve enough (Δ {target_metric_key}: {metric_difference:+.4f}). Sticking with deployed model."

        # --- 4. Log and Return ---

        logging.info(f"\n{'='*60}")
        logging.info(f"📊 EVALUATION SUMMARY:")
        logging.info(f"✅ New Model {target_metric_key}: {new_metric_value:.4f}")
        logging.info(
            f"✅ Deployed Model {target_metric_key}: {deployed_metric_value:.4f}"
        )
        logging.info(f"✅ Difference (New - Deployed): {metric_difference:+.4f}")
        logging.info(
            f"\n🎯 DECISION: {'✅ PROMOTE' if should_promote else '❌ DO NOT PROMOTE'}"
        )
        logging.info(f"✅ Reason: {reason}")
        logging.info(f"{'='*60}\n")

        promotion_status = "True" if should_promote else "False"

        # Log Metrics
        evaluation_metrics.log_metric(
            f"new_model_{target_metric_key}", float(round(new_metric_value, 4))
        )
        evaluation_metrics.log_metric(
            f"deployed_model_{target_metric_key}",
            float(
                round(
                    (
                        deployed_metric_value
                        if deployed_metric_value != 10000000
                        else -1.0
                    ),
                    4,
                )
            ),
        )
        evaluation_metrics.log_metric(
            f"{target_metric_key}_difference", float(round(metric_difference, 4))
        )
        evaluation_metrics.log_metric("should_promote", 1.0 if should_promote else 0.0)

        # Return output
        return outputs(promotion_status)

    except Exception as e:
        logging.error(f"❌ Critical pipeline failure: {e}")
        logging.error(traceback.format_exc())
        return default_failure_output
