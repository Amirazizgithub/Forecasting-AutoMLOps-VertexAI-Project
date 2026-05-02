# ==============================================================================
# CHECK OR CREATE CLOUD STORAGE BUCKET TASK
# ==============================================================================

"""
GCS Bucket Check and Creation Component

This component checks if a GCS bucket exists and creates it if it doesn't.
"""

# components/check_bucket.py
from typing import NamedTuple
from kfp.dsl import component, Output, Metrics


@component(
    packages_to_install=["google-cloud-storage", "google-api-core"],
    base_image="python:3.10",
)
def check_and_create_gcs_bucket(
    bucket_name: str,
    project_id: str,
    region: str,
    bucket_details: Output[Metrics],
) -> NamedTuple("Outputs", [("bucket_status", bool)]):  # type: ignore
    """
    KFP component to check if a Google Cloud Storage (GCS) bucket exists.
    If it does not exist, it attempts to create it at the specified location.
    """
    from collections import namedtuple
    from google.cloud import storage
    import logging
    from google.api_core import exceptions

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.info(f"✅ {25*'#'} CHECK OR CREATE CLOUD STORAGE BUCKET TASK {25*'#'}")
    try:
        bucket_status = False
        logging.info(f"✅ Starting GCS bucket management for '{bucket_name}'...")
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(bucket_name)
        bucket_location = bucket.location

        if bucket.exists():
            if bucket_location:
                logging.info(
                    f"✅ GCS bucket: '{bucket_name}' already exists in location: {bucket_location}. Proceeding."
                )
            else:
                logging.info(
                    f"✅ GCS bucket: '{bucket_name}' already exists in location: unknown/multi-region. Proceeding."
                )
            bucket_status = True
        else:
            # Only try to create if bucket doesn't exist
            logging.info(
                f"⏳ GCS bucket '{bucket_name}' does not exist. Attempting to create it in location '{region}'..."
            )
            new_bucket = storage_client.create_bucket(
                bucket_name, project=project_id, location=region
            )
            bucket_status = True
            logging.info(
                f"✅ GCS bucket '{bucket_name}' created successfully in {new_bucket.location}."
            )

        # Return bucket status as bool
        logging.info(f"✅ Bucket Status: {bucket_status}")

        # Bucket details logs
        bucket_details.log_metric("project_id", project_id)
        bucket_details.log_metric("region", region)
        bucket_details.log_metric("bucket_name", bucket_name)
        bucket_details.log_metric("bucket_status", 1.0 if bucket_status else 0.0)

        # Create named tuple for output
        outputs = namedtuple("Outputs", ["bucket_status"])
        return outputs(bucket_status)

    except exceptions.Forbidden as e:
        logging.error(
            f"❌ Permission denied to create bucket '{bucket_name}'. Ensure service account has 'storage.buckets.create'."
        )
        logging.error(f"❌ Deployment failure due to GCS permission error: {e}")
        raise RuntimeError(f"❌ Deployment failure due to GCS permission error: {e}")

    except exceptions.GoogleAPICallError as e:
        logging.error(
            f"❌ An unexpected error occurred while managing bucket '{bucket_name}': {e}"
        )
        logging.error(f"❌ GCS bucket operation failed: {e}")
        raise RuntimeError(f"❌ GCS bucket operation failed: {e}")
