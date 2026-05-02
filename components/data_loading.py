# ==============================================================================
# TIME SERIES DATA LOADING & CREATING DATASET TASK
# ==============================================================================

"""
Data Loading Component

This component loads data and creates a Vertex AI dataset.
"""

# components/data_loading.py
from kfp.dsl import component, Output, Dataset, Metrics


@component(
    packages_to_install=[
        "pandas",
        "numpy",
        "google-cloud-bigquery",
        "google-cloud-bigquery-storage",
        "google-api-core",
        "db-dtypes",
    ],
    base_image="python:3.10",
)
def load_and_create_dataset_component(
    project_id: str,
    sql_query: str,
    bucket_status: bool,
    training_data: Output[Dataset],
    data_metrics: Output[Metrics],
):
    """Loads and creates data from BigQuery for preprocessing."""
    import pandas as pd
    from google.cloud import bigquery
    import logging
    from google.api_core import exceptions

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info(
        f"✅ {25*'#'} TIME SERIES DATA LOADING & CREATING DATASET TASK {25*'#'}"
    )

    if bucket_status != True:
        print(
            "Bucket check failed or bucket was not created/found. Exiting data loading."
        )
        logging.error(
            "❌ Bucket check failed or bucket was not created/found. Exiting data loading."
        )
        raise RuntimeError(
            f"❌ Bucket check failed or bucket was not created/found: {e}"
        )

    logging.info(f"✅ Loading data from {project_id}")

    # Load data from big query in dataframe
    bq_client = bigquery.Client(project=project_id)
    logging.info("✅ Executing query to load data from BigQuery...")
    try:
        df = bq_client.query(sql_query).to_dataframe()
        # Log basic metrics as scalars (not complex types)
        data_metrics.log_metric("total_rows", int(df.shape[0]))
        data_metrics.log_metric("total_columns", int(df.shape[1]))
    except exceptions.GoogleAPICallError as e:
        logging.error(f"❌ Failed to load data from BigQuery: {e}")
        raise RuntimeError(f"❌ Failed to load data from BigQuery: {e}")
    logging.info("✅ Data loaded from BigQuery successfully.")

    try:
        logging.info(f"✅ Dataframe shape before preprocessing: {df.shape}")
        # --- Simple Preprocessing (from notebook) ---
        # Convert potential datetime columns
        for col in df.columns:
            if df[col].dtype == "object":
                # Heuristic to check if column can be coerced to datetime
                sample = df[col].dropna().astype(str).head(100)
                success_rate = sum(
                    1
                    for val in sample
                    if pd.to_datetime(val, errors="coerce") is not pd.NaT
                ) / (len(sample) or 1)
                if success_rate >= 0.8:
                    df[col] = pd.to_datetime(df[col], errors="coerce")

        # Update feature roles
        numerical_cols = [
            col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])
        ]
        categorical_cols = [
            col for col in df.columns if pd.api.types.is_string_dtype(df[col])
        ]
        date_cols = [
            col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])
        ]
        # Sort by date
        if date_cols:
            col = date_cols[0]
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df = df.sort_values(by=col, ascending=True).reset_index(drop=True)

        # Log metrics as scalars only (counts)
        data_metrics.log_metric("num_numerical_columns", len(numerical_cols))
        data_metrics.log_metric("num_categorical_columns", len(categorical_cols))
        data_metrics.log_metric("num_date_columns", len(date_cols))
        logging.info(f"✅ Numerical columns count: {len(numerical_cols)}")
        logging.info(f"✅ Categorical columns count: {len(categorical_cols)}")
        logging.info(f"✅ Date columns count: {len(date_cols)}")

        # Handle missing values
        # Replace None with np.nan for compatibility
        null_counts = int(df.isnull().sum().sum())
        data_metrics.log_metric("total_null_values", null_counts)
        logging.info(f"✅ Total null values before imputation: {null_counts}")

        # Remove duplicates
        logging.info("✅ Handling duplicates in the dataframe...")
        duplicate_count = int(df.duplicated().sum())
        data_metrics.log_metric("duplicate_rows", duplicate_count)
        logging.info(f"✅ Duplicate rows found: {duplicate_count}")
        df = df.drop_duplicates(ignore_index=True).reset_index(drop=True)

        # Log final metrics
        data_metrics.log_metric("final_row_count", int(len(df)))
        logging.info(f"✅ Data loaded successfully with {len(df)} rows.")

        # Save the processed data artifact
        df.to_csv(training_data.path, index=False)
        logging.info(f"✅ Processed data saved to artifact path: {training_data.path}")

    except exceptions.GoogleAPICallError as e:
        logging.error(f"❌ Data preprocessing failed: {e}")
        raise RuntimeError(f"❌ Data preprocessing failed: {e}")
