# ==============================================================================
# TIME SERIES FORECASTING SCRIPT
# ==============================================================================

"""
Forecasting Pipeline Module
This module defines the ForecastingPipeline class which handles loading the latest
trained time series model from Vertex AI Model Registry and making *future* forecasts.
"""

import os
import json
import pandas as pd
import joblib
import logging
import tempfile
from datetime import date, timedelta
from pipeline_config import Pipeline_Config
from google.cloud import aiplatform, storage
from google.api_core import exceptions
from google.cloud import bigquery

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ForecastingPipeline(Pipeline_Config):
    def __init__(self, config_data: dict) -> None:
        """
        config_data: dict with PROJECT_ID, REGION, MODEL_DISPLAY_NAME, etc.
        """
        try:
            logging.info(
                "✅ Initializing ForecastingPipeline with provided config data."
            )
            # Initialize parent class first to set up all config attributes
            super().__init__(data=config_data)
            # Now load metadata after config is initialized
            self.metadata = self.load_metadata_from_gcs()
            self.last_train_date = self.metadata.get(
                "last_train_date", date.today().strftime("%Y-%m-%d")
            )
            self.forecast_start_date = (
                date.fromisoformat(self.last_train_date) + timedelta(days=1)
            ).strftime("%Y-%m-%d")
            logging.info(f"✅ Last training date from metadata: {self.last_train_date}")
            logging.info(f"✅ Forecast start date set to: {self.forecast_start_date}")
        except Exception as e:
            logging.error(f"❌ Error during ForecastingPipeline initialization: {e}")
            raise RuntimeError(
                f"❌ Error during ForecastingPipeline initialization: {e}"
            )

    def setup_authentication(self) -> None:
        """Setup GCP authentication from service account JSON if available."""
        if getattr(self, "SERVICE_ACCOUNT_KEY", None) and os.path.exists(
            self.SERVICE_ACCOUNT_KEY
        ):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.SERVICE_ACCOUNT_KEY
            logging.info(f"✅ Using service account: {self.SERVICE_ACCOUNT_KEY}")
        else:
            logging.info(
                "ℹ️ Using default ADC. Ensure gcloud auth application-default login or workload identity."
            )

    def _download_model_file(self, artifact_uri: str, local_dir: str) -> str:
        """Download model.pkl or model.joblib from model.artifact_uri (gs://...). Returns local file path."""
        if not artifact_uri or not artifact_uri.startswith("gs://"):
            raise ValueError(f"Invalid artifact_uri: {artifact_uri}")

        # Parse GCS URI
        gcs_path = artifact_uri.replace("gs://", "")
        bucket_name = gcs_path.split("/")[0]
        blob_prefix = "/".join(gcs_path.split("/")[1:]).rstrip("/")

        client = storage.Client(project=self.PROJECT_ID)
        bucket = client.bucket(bucket_name)

        model_filename = "model.pkl"
        blob_path = f"{blob_prefix}/{model_filename}" if blob_prefix else model_filename
        blob = bucket.blob(blob_path)
        if blob.exists():
            local_path = os.path.join(local_dir, model_filename)
            blob.download_to_filename(local_path)
            logging.info(
                f"✅ Downloaded model file: gs://{bucket_name}/{blob_path} -> {local_path}"
            )
            return local_path

        raise FileNotFoundError(
            f"❌ model.pkl not found under gs://{bucket_name}/{blob_prefix}/"
        )

    def get_latest_model_from_registry(self) -> aiplatform.Model:
        """Get latest Vertex AI Model resource by display_name."""
        self.setup_authentication()
        aiplatform.init(project=self.PROJECT_ID, location=self.REGION)

        models = aiplatform.Model.list(
            filter=f'display_name="{self.MODEL_DISPLAY_NAME}"',
            order_by="create_time desc",
        )
        if not models:
            raise ValueError(
                f"❌ No forecasting model found with display_name: {self.MODEL_DISPLAY_NAME}"
            )

        latest = models[0]
        logging.info(f"✅ Found latest forecasting model: {latest.resource_name}")
        logging.info(f"✅ Artifact URI: {latest.uri}")
        return latest

    def load_metadata_from_gcs(self) -> dict | None:
        """
        Downloads a JSON file from Google Cloud Storage and loads it into a dictionary.

        Args:
            gcs_url: The full gs:// path to the JSON file.
            service_account_key: The path to the local service account key file.

        Returns:
            A dictionary containing the loaded JSON data, or None on failure.
        """
        logging.info(f"✅ Attempting to load JSON from GCS: {self.MODEL_METADATA_PATH}")

        try:
            # 1. Setup Authentication
            self.setup_authentication()

            # 2. Initialize a client (will use the credentials set above)
            logging.info("✅ Starting storage client initialization...")
            storage_client = storage.Client()
            logging.info("✅ Client successfully initialized.")

            # 3. Get the bucket and the specific blob (file)
            bucket = storage_client.bucket(self.BUCKET_NAME)
            blob = bucket.blob(self.MODEL_METADATA_PATH)

            # 4. Download the file contents as a string
            json_string = blob.download_as_text()
            logging.info(f"✅ Successfully downloaded file: {self.MODEL_METADATA_PATH}")

            # 5. Load the JSON string into a Python dictionary
            metadata = json.loads(json_string)
            logging.info(f"✅ Loaded metadata from gcs. Metadata: {metadata}")
            return metadata

        except Exception as e:
            # Catch common GCS errors (e.g., file not found, permission issues)
            logging.error(f"❌ Failed to load JSON from GCS: {e}")
            raise (f"❌ Failed to load JSON from GCS: {e}")

    def forecast(
        self,
        periods: int,
        exog_features_df: pd.DataFrame = None,
        date_index_col: str = "date",
    ) -> pd.DataFrame:
        """
        Load latest time series model artifact and predict future values.

        Args:
            periods (int): The number of future time steps (e.g., days, months) to forecast.
            exog_features_df (pd.DataFrame, optional): DataFrame containing future
                exogenous features required by the model (e.g., future prices, promotions).
                Must have 'periods' rows. Defaults to None.
            date_index_col (str): The column name used for the date index (e.g., 'ds').

        Returns:
            pd.DataFrame: DataFrame with future dates as index and 'forecast' column.
        """
        try:
            model = self.get_latest_model_from_registry()
            artifact_uri = model.uri

            if exog_features_df is not None:
                if len(exog_features_df) != periods:
                    raise ValueError(
                        f"❌ Exogenous features DataFrame must have {periods} rows, found {len(exog_features_df)}."
                    )
                logging.info(f"✅ Using {periods} steps of exogenous features.")

            with tempfile.TemporaryDirectory() as local_dir:
                local_model_path = self._download_model_file(artifact_uri, local_dir)
                model_object = joblib.load(local_model_path)

            # --- Forecasting Logic based on Model Type ---

            # 1. Attempt to handle Prophet model (has make_future_dataframe method)
            if hasattr(model_object, "make_future_dataframe"):
                logging.info(
                    f"🔮 Model identified as Prophet. Generating future dates..."
                )

                # Prophet needs the entire date range, even if it's only for the future
                future = model_object.make_future_dataframe(
                    periods=periods, include_history=False
                )

                # Check for exogenous features and merge them
                if exog_features_df is not None:
                    # Rename the date column in exog_features_df to 'ds' for Prophet compatibility
                    if date_index_col in exog_features_df.columns:
                        exog_features_df = exog_features_df.rename(
                            columns={date_index_col: "ds"}
                        )

                    # Ensure the date column is used for merging (Prophet requires 'ds')
                    future = pd.merge(future, exog_features_df, how="left", on="ds")

                preds_df = model_object.predict(future)

                # Output DataFrame indexed by date
                results_df = preds_df.set_index("ds")[["yhat"]].rename(
                    columns={"ds": "date", "yhat": "forecast"}
                )
                results_df.index.name = date_index_col

            # 2. Attempt to handle Statsmodels/pmdarima (ARIMA/SARIMAX/ETS) models
            elif hasattr(model_object, "forecast") or hasattr(
                model_object, "get_forecast"
            ):
                logging.info(
                    f"📈 Model identified as Statsmodels/pmdarima (ARIMA/SARIMAX/ETS)."
                )

                # Statsmodels models require exog as a DataFrame or NumPy array
                exog_array = None
                if exog_features_df is not None:
                    exog_array = exog_features_df.values

                # Use get_forecast for statsmodels (SARIMAX, ARIMA) - provides confidence intervals
                if hasattr(model_object, "get_forecast"):
                    logging.info(
                        f"📊 Using get_forecast() method for prediction with confidence intervals."
                    )
                    forecast_result = model_object.get_forecast(
                        steps=periods, exog=exog_array
                    )
                    preds = forecast_result.predicted_mean
                    conf_int = forecast_result.conf_int(alpha=0.05)  # 95% CI

                    # Create future dates
                    future_dates = (
                        pd.date_range(
                            start=self.forecast_start_date, periods=periods, freq="D"
                        )
                        .normalize()
                        .strftime("%Y-%m-%d")
                    )

                    results_df = pd.DataFrame(
                        {"forecast": preds.values}, index=future_dates
                    )
                    results_df.index.name = date_index_col

                # Fallback to forecast() method (pmdarima, older statsmodels)
                else:
                    logging.info(f"📊 Using forecast() method for prediction.")
                    forecast_result = model_object.forecast(
                        steps=periods, exogenous=exog_array
                    )

                    # Forecast result can be a Series (ETS) or tuple/object (ARIMA)
                    if isinstance(forecast_result, tuple):
                        preds = forecast_result[0]
                    else:
                        preds = forecast_result

                    # Create future dates
                    future_dates = (
                        pd.date_range(
                            start=self.forecast_start_date, periods=periods, freq="D"
                        )
                        .normalize()
                        .strftime("%Y-%m-%d")
                    )

                    results_df = pd.DataFrame({"forecast": preds}, index=future_dates)
                    results_df.index.name = date_index_col

            else:
                raise TypeError(
                    f"❌ Model of type {type(model_object)} is not recognized for forecasting (requires .predict or .forecast method)."
                )

            logging.info(
                f"✅ Forecasting completed. {len(results_df)} periods predicted."
            )

            return results_df

        except exceptions.GoogleAPICallError as e:
            logging.error(f"❌ Error accessing Vertex AI or GCS: {e}")
            raise RuntimeError(f"Error accessing Vertex AI or GCS: {e}")
        except Exception as e:
            logging.error(f"❌ Forecasting failed: {e}")
            raise RuntimeError(f"❌ Forecasting failed: {e}")

    def load_inventory_data_from_bigquery(self, start_data: str) -> pd.DataFrame:
        self.setup_authentication()
        try:
            bigquery_client = bigquery.Client(project=self.PROJECT_ID, location="US")
            query = f"""
                SELECT
                date,
                COALESCE(SUM(inventory), 0) AS total_inventory
                FROM
                `doreamon-1752016732628.Unicommerce.inventory`
                WHERE
                itemTypeSKU = '{self.ITEMSKU}'
                AND (date BETWEEN DATE '{start_data}'
                AND CURRENT_DATE()+1)
                GROUP BY
                1
                ORDER BY
                date ASC;
            """

            inventory_df = bigquery_client.query(query=query).to_dataframe()
            logging.info(
                f"✅ Loaded inventory data from BigQuery starting from {start_data}."
            )
            return inventory_df
        except exceptions.GoogleAPICallError as e:
            logging.error(f"❌ Error accessing BigQuery: {e}")
            raise RuntimeError(f"❌ Error accessing BigQuery: {e}")
        except Exception as e:
            logging.error(f"❌ Failed to load inventory data: {e}")
            raise RuntimeError(f"❌ Failed to load inventory data: {e}")

    def final_inventory_consumption_data(
        self, forecast_df: pd.DataFrame
    ) -> pd.DataFrame:
        try:
            # Ensure forecast_df date column is also datetime type
            if "date" in forecast_df.columns:
                forecast_df["date"] = pd.to_datetime(forecast_df["date"])
            else:
                raise ValueError("❌ 'date' column not found in forecast_df")

            # Rename 'forecast' column to 'predicted_unit_sold' if it exists
            if "forecast" in forecast_df.columns:
                forecast_df = forecast_df.rename(
                    columns={"forecast": "predicted_unit_sold"}
                )
            elif "predicted_unit_sold" not in forecast_df.columns:
                raise ValueError(
                    "❌ Neither 'forecast' nor 'predicted_unit_sold' column found in forecast_df"
                )

                # # Load the inventory dataframe from big query
                # start_date = forecast_df['date'].iloc[0]
                # # Convert Timestamp to string format for BigQuery DATE literal
                # if isinstance(start_date, pd.Timestamp):
                #     start_date_str = start_date.strftime('%Y-%m-%d')
                # else:
                #     start_date_str = str(start_date)

            inventory_df = self.load_inventory_data_from_bigquery(
                start_data=self.forecast_start_date
            )
            inventory_df["date"] = pd.to_datetime(inventory_df["date"])
            logging.info("✅ Successfully loaded inventory data for merging.")

            # join df_forecast and df_inv on date
            df_merged = pd.merge(
                forecast_df, inventory_df, left_on="date", right_on="date", how="left"
            )
            logging.info("🔍 Merged forecast data with inventory data.")

            # Data Cleaning and Preparation
            df = df_merged.copy()
            df["date"] = pd.to_datetime(df["date"])
            # Crucially sort by date to ensure the sequential calculation is correct
            df = df.sort_values(by="date").reset_index(drop=True)

            # Convert numeric columns to float to avoid Int64 type issues during calculations
            if "total_inventory" in df.columns:
                df["total_inventory"] = df["total_inventory"].astype(float)
            if "predicted_unit_sold" in df.columns:
                df["predicted_unit_sold"] = df["predicted_unit_sold"].astype(float)

            # 2. INITIALIZE THE NEW COLUMN
            # We use the existing 'total_inventory' values for days where we have actual data
            df["inventory_consumption"] = df["total_inventory"].astype(float)

            # 3. SEQUENTIAL INVENTORY PROJECTION LOGIC
            # Find the index of the first row where 'total_inventory' is null (start of forecast)
            nan_start_index = df["total_inventory"].isna().idxmax()

            # Check if we actually have a forecast period to calculate
            if nan_start_index < len(df) and pd.isna(
                df.loc[nan_start_index, "total_inventory"]
            ):

                # Check if there's historical data (nan_start_index > 0)
                if nan_start_index > 0:
                    # Get the inventory value from the day *before* the forecast starts
                    last_known_inventory = df.loc[
                        nan_start_index - 1, "total_inventory"
                    ]
                else:
                    # No historical data available - start with 0 or a default value
                    last_known_inventory = 0
                    logging.warning(
                        "⚠️ No historical inventory data found. Starting with 0 inventory."
                    )

                # Iterate through the forecast period (from nan_start_index to the end)
                for i in range(nan_start_index, len(df)):

                    current_predicted_sales = df.loc[i, "predicted_unit_sold"]

                    if i == nan_start_index:
                        # For the first forecast day, use the last known inventory
                        previous_inventory = last_known_inventory
                    else:
                        # For subsequent days, use the projected inventory from the previous day (i-1)
                        previous_inventory = df.loc[i - 1, "inventory_consumption"]

                    # Calculate the new inventory level (Current Inventory = Previous Inventory - Predicted Sales)
                    df.loc[i, "inventory_consumption"] = (
                        previous_inventory - current_predicted_sales
                    )

            # 4. FINAL CLEANUP AND OUTPUT
            # Handle NaN, Infinity, and other non-JSON-compliant values
            # Replace NaN, inf, -inf with None (which becomes null in JSON)
            df = df.replace([float("inf"), float("-inf"), float("nan")], None)
            df = df.fillna(value=0)  # Convert NaN to None
            # Replace negative values in 'predicted_unit_sold' with 0
            df["predicted_unit_sold"] = (
                df["predicted_unit_sold"].clip(lower=0).round().astype(int)
            )

            # 5. Convert df in integer
            df["inventory_consumption"] = df["inventory_consumption"].astype(int)
            df["predicted_unit_sold"] = df["predicted_unit_sold"].astype(int)
            df["total_inventory"] = df["total_inventory"].astype(int)

            logging.info(
                f"✅ Successfully merged the forecast data with inventory data. Dataframe shape: {df.shape}"
            )
            return df

        except exceptions.GoogleAPICallError as e:
            logging.error(
                f"❌ Error to merged the forecast data with inventory data: {e}"
            )
            raise RuntimeError(
                f"❌ Error to merged the forecast data with inventory data: {e}"
            )
        except Exception as e:
            logging.error(
                f"❌ Failed to merge the forecast data with inventory data: {e}"
            )
            raise RuntimeError(
                f"❌ Failed to merge the forecast data with inventory data: {e}"
            )
