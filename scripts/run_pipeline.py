# ==============================================================================
# PIPELINE RUNNER SCRIPT
# ==============================================================================

"""
Pipeline Runner Script

This script compiles and runs the Vertex AI MLOps pipeline.
It provides functionality to compile, submit, and monitor pipeline runs.
"""

# scripts/run_pipeline.py
import sys
import os
import logging
from google.api_core import exceptions

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
from datetime import datetime
from kfp.compiler import Compiler
from google.cloud import aiplatform

# Import configuration and pipeline function
from pipeline_config import Pipeline_Config


class Compile_Run_Pipeline(Pipeline_Config):
    def __init__(self, data: dict):
        super().__init__(data=data)

        """Compiles the KFP pipeline and runs it on Vertex AI."""
        logging.info(f"✅ {25*'#'} PIPELINE COMPILATION AND RUN {25*'#'}")

    def compile_pipeline(self, pipeline_func):
        try:
            # Compile the pipeline
            logging.info(f"Compiling pipeline to: {self.PIPELINE_PACKAGE_PATH}")
            Compiler().compile(
                pipeline_func=pipeline_func,
                package_path=self.PIPELINE_PACKAGE_PATH,
            )
            logging.info("✅ Pipeline compilation successful.")
        except exceptions.GoogleAPICallError as e:
            logging.error(f"❌ Error during pipeline compilation: {e}")
            raise RuntimeError(f"❌ Error during pipeline compilation: {e}")

    def run_pipeline(self, parameter_overrides: dict | None = None):
        try:
            # 0. Set up authentication
            # Set the path to your service account key file
            service_account_path = self.SERVICE_ACCOUNT_KEY
            if os.path.exists(service_account_path):
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path
                logging.info(
                    f"✅ Using service account credentials from: {service_account_path}"
                )
            else:
                logging.warning(
                    f"⚠️ Service account key file not found at: {service_account_path}"
                )
                logging.warning("⚠️ Attempting to use default credentials...")

            # 1. Initialize Vertex AI SDK
            aiplatform.init(project=self.PROJECT_ID, location=self.REGION)

            # 2. Define runtime parameters
            TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
            # Convert to lowercase and replace invalid characters for Vertex AI job_id requirements
            item_sku_clean = self.ITEMSKU.lower().replace("_", "-")
            job_id = f"forecast-{item_sku_clean}-{TIMESTAMP}"

            parameter_values = {
                "project_id": self.PROJECT_ID,
                "region": self.REGION,
                "bucket_name": self.BUCKET_NAME,
                "sql_query": self._SQL_QUERY.replace("\n", " ").replace("\r", ""),
                "dataset_display_name": self.DATASET_DISPLAY_NAME,
                "target_variable": self.TARGET_VARIABLE,
                "date_variable": self.DATE_VARIABLE,
                "serving_container_image_uri": self.SERVING_CONTAINER_IMAGE_URI,
                "model_display_name": self.MODEL_DISPLAY_NAME,
                "endpoint_display_name": self.ENDPOINT_DISPLAY_NAME,
                "model_deployment_path": self.MODEL_DEPLOYMENT_PATH,
                "model_version": self.MODEL_VERSION,
            }

            if parameter_overrides:
                parameter_values.update(parameter_overrides)

            # 3. Run the pipeline job
            logging.info(f"✅ Submitting pipeline job: {job_id}")
            pipeline_job = aiplatform.PipelineJob(
                display_name=self.MODEL_DISPLAY_NAME,
                pipeline_root=self.PIPELINE_ROOT,
                template_path=self.PIPELINE_PACKAGE_PATH,
                job_id=job_id,
                parameter_values=parameter_values,
                enable_caching=True,
            )

            # Note: Running synchronously here for simplicity, but asynchronous is typical.
            # The 'run()' method blocks until completion.
            pipeline_job.run()
            logging.info(f"✅ Pipeline execution successfully")
            logging.info("✅ \n--- Pipeline Job Status ---")
            logging.info("✅ Resource: %s", pipeline_job.resource_name)
            logging.info("✅ State: %s", pipeline_job.state)
        except exceptions.GoogleAPICallError as e:
            logging.error(f"❌ Error during pipeline execution: {e}")
            sys.exit(1)
            raise RuntimeError(f"❌ Error during pipeline execution: {e}")
