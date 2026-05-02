# ==============================================================================
# PIPELINE CONFIGURATION SETTINGS
# ==============================================================================

"""
Configuration settings for Vertex AI Forecasting MLOps Pipeline.

This module contains all the configuration variables and settings
used throughout the pipeline components and scripts.
"""


class Pipeline_Config:
    def __init__(self, data: dict) -> None:
        try:
            # --- Google Cloud Configuration ---
            self.ROUTES_SUFFIX: str = data.get(
                "ROUTES_SUFFIX", "revenue_forecast"
            )  # Suffix for API routes
            self.CLIENT_NUMBER: int = data.get("CLIENT_NUMBER")
            self.START_DATE: str = data.get("START_DATE", "2020-01-01")
            self.END_DATE: str = data.get("END_DATE", "2026-06-30")
            self.SOURCE_NAME: str = (
                data.get("SOURCE_NAME", "unknown")
                .replace(" ", "_")
                .replace(",", "_")
                .lower()
            )
            self.ITEMSKU: str = data.get("ITEMSKU", "")
            self.TARGET_VARIABLE: str = (
                data.get("TARGET_VARIABLE", "unknown")
                .replace(" ", "_")
                .replace(",", "_")
                .lower()
            )
            self.DATE_VARIABLE: str = data.get("DATE_VARIABLE", "Date")
            self.MODEL_PIPELINE: str = (
                data.get("MODEL_PIPELINE", "default_pipeline")
                .replace(" ", "_")
                .replace(",", "_")
                .lower()
            )
            self.PROJECT_ID: str = data.get("PROJECT_ID")
            self.TABLE_ID: str = data.get("TABLE_ID")
            self.REGION: str = data.get("CLIENT_REGION", "us-central1")
            self.SERVICE_ACCOUNT_KEY = "gcp_project_creds.json"
            self.BUCKET_NAME: str = (
                f"forecasting-{self.CLIENT_NUMBER}-{self.SOURCE_NAME}-bucket"
            )
        except KeyError as e:
            raise KeyError(f"Missing required configuration key: {e}")
        except Exception as e:
            raise Exception(f"Error initializing Pipeline_Config: {e}")

        # --- Data Configuration ---
        self.DATASET_DISPLAY_NAME: str = (
            f"{self.MODEL_PIPELINE}-{self.CLIENT_NUMBER}-{self.SOURCE_NAME}-{self.TARGET_VARIABLE}-forecast-dataset"
        )

        # --- MLOps Pipeline Configuration ---
        self.PIPELINE_ROOT: str = (
            f"gs://{self.BUCKET_NAME}/pipeline_root_{self.MODEL_PIPELINE}_{self.TARGET_VARIABLE}_model"
        )
        self.PIPELINE_NAME: str = (
            f"{self.MODEL_PIPELINE}-{self.CLIENT_NUMBER}-{self.SOURCE_NAME}-{self.TARGET_VARIABLE}-forecasting-pipeline"
        )
        self.PIPELINE_PACKAGE_PATH: str = (
            f"vertexai-job-pipelines/{self.MODEL_PIPELINE}-{self.CLIENT_NUMBER}-{self.SOURCE_NAME}-{self.TARGET_VARIABLE}-forecasting-pipeline.yaml"
        )

        # --- Model & Endpoint Configuration ---
        self.MODEL_DEPLOYMENT_PATH: str = (
            f"pipeline_root_{self.MODEL_PIPELINE}_{self.TARGET_VARIABLE}_model/deployed_models/{self.TARGET_VARIABLE}_forecasting"
        )
        self.MODEL_VERSION: str = "v1"
        self.MODEL_DISPLAY_NAME: str = (
            f"{self.MODEL_PIPELINE}-{self.CLIENT_NUMBER}-{self.SOURCE_NAME}-{self.TARGET_VARIABLE}-forecast-model"
        )
        self.MODEL_METADATA_PATH: str = (
            f"{self.MODEL_DEPLOYMENT_PATH}/model_metadata.json"
        )
        self.ENDPOINT_DISPLAY_NAME: str = (
            f"{self.MODEL_PIPELINE}-{self.CLIENT_NUMBER}-{self.SOURCE_NAME}-{self.TARGET_VARIABLE}-forecast"
        )

        # --- Serving Container Configuration ---
        self.SKLEARN_VERSION: str = "1.5.0"
        # Use public Google Cloud pre-built container instead of Artifact Registry
        # This avoids permission issues with Artifact Registry
        # SERVING_CONTAINER_IMAGE_URI: str = f'{REGION}-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.{SKLEARN_VERSION}:{MODEL_VERSION}'
        # SERVING_CONTAINER_IMAGE_URI: str = f"us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.{SKLEARN_VERSION}:latest"
        self.SERVING_CONTAINER_IMAGE_URI = (
            "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest"
        )

        # --- SQL Queries ---
        # IMPORTANT:
        # - Keep only ASCII spaces, no U+00A0 or zero-width spaces.
        # - Do not indent with tabs.
        # - Backtick fully qualified tables to avoid accidental parsing issues on some serializers.

        self._REVENUE_BASE_SQL = """
          WITH
            all_date AS (
              SELECT * FROM UNNEST(generate_date_array('{start_date}', '{end_date}'))
                AS new_date
            ),
            add_stats AS (
              SELECT
                CAST(date(new_date) AS STRING) {date_variable},
                ROUND(COALESCE(sum(Impressions), 0), 2) AS Impressions,
                ROUND(COALESCE(sum(Conversions), 0), 2) AS Conversions,
                ROUND(COALESCE(sum(Clicks), 0), 2) AS Clicks,
                ROUND(COALESCE(sum(Ad_Spend_INR), 0), 2) AS Ad_Spend,
                ROUND(COALESCE(sum(Revenue_INR), 0), 2) AS {target_variable}
              FROM all_date ad
              LEFT JOIN {table_id} pt
                ON ad.new_date = (pt.date)
              WHERE new_date BETWEEN '{start_date}' AND '{end_date}'
              GROUP BY 1
            )
          SELECT * FROM add_stats ORDER BY {date_variable} ASC;
          """
        # Final SQL with clean formatting
        self.REVENUE_SQL_QUERY: str = self._REVENUE_BASE_SQL.format(
            start_date=self.START_DATE,
            end_date=self.END_DATE,
            table_id=self.TABLE_ID,
            date_variable=self.DATE_VARIABLE,
            target_variable=self.TARGET_VARIABLE,
        )

        if self.ROUTES_SUFFIX == "revenue_forecast":
            self._SQL_QUERY = self.REVENUE_SQL_QUERY
