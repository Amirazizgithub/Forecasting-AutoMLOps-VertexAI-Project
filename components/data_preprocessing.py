# ==============================================================================
# TIME SERIES DATA PREPROCESSING FOR MODEL TRAINING TASK
# ==============================================================================

"""
Data Preprocessing Component

This component preprocesses the data for machine learning.
"""

# components/preprocessing.py
from kfp.dsl import component, Input, Output, Dataset, Metrics
from typing import List, Tuple, Union


@component(
    packages_to_install=[
        "pandas",
        "numpy",
        "google-cloud-aiplatform",
        "google-api-core",
        "scikit-learn",
    ],
    base_image="python:3.10",
)
def data_preprocessing_component(
    project_id: str,
    region: str,
    date_variable: str,
    target_variable: str,
    dataset_display_name: str,
    training_data: Input[Dataset],
    preprocess_data: Output[Dataset],
    correlation_data: Output[Dataset],
    preprocess_data_metrics: Output[Metrics],
    interpolation_method: str = "time",  # e.g., 'time', 'linear', 'spline'
    clip_target_quantile: str = "0.01,0.99",
    drop_id_like: bool = True,
    id_like_keywords: List[str] = ["id", "_id", "uuid", "guid"],
    min_non_null_ratio: float = 0.80,
    categorical_limit: int = 10,
    rare_category_threshold: float = 0.01,
    quasi_constant_threshold: float = 0.90,
    correlation_threshold: float = 0.20,
):
    """
    Robust preprocessing for time series forecasting, including:
    - Date-Time feature engineering (year, month, dayofweek, etc.).
    - Target variable clipping for outlier reduction.
    - Interpolation-based NaN handling for the target variable.
    - Standard feature selection (ID-like, high-null, quasi-constant).
    - Handling of categorical features (high cardinality, rare categories).
    - All non-date/target columns become exogenous features.
    """

    import pandas as pd
    import numpy as np
    import logging
    from google.api_core import exceptions
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import VarianceThreshold
    from google.cloud import aiplatform

    # --- Setup ---
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info(f"{25*'#'} TIME SERIES DATA PREPROCESSING TASK {25*'#'}")
    target = target_variable
    date_col = date_variable

    # 🔑 KFP FIX: Parse the string back into a Tuple[float, float]
    try:
        if clip_target_quantile:
            q_list = [float(q.strip()) for q in clip_target_quantile.split(",")]
            if len(q_list) == 2:
                clip_q_tuple: Union[Tuple[float, float], None] = (q_list[0], q_list[1])
            else:
                clip_q_tuple = None
                logging.warning(
                    "⚠️ clip_target_quantile must be two comma-separated floats. Clipping disabled."
                )
        else:
            clip_q_tuple = None
    except Exception:
        clip_q_tuple = None
        logging.error("❌ Failed to parse clip_target_quantile. Clipping disabled.")

    # ------------------ 1. Data Loading and Initial Checks ------------------
    try:
        df = pd.read_csv(training_data.path)
        logging.info(f"✅ Successfully loaded data. Dataframe Shape: {df.shape}")
        rows0, cols0 = df.shape
        preprocess_data_metrics.log_metric("total_rows_initial", float(rows0))
        preprocess_data_metrics.log_metric("total_columns_initial", float(cols0))

        # Basic checks
        if target not in df.columns or date_col not in df.columns:
            raise RuntimeError("❌ Target or Date variable not found.")
        if not pd.api.types.is_numeric_dtype(df[target]):
            raise RuntimeError(f"❌ Target column '{target}' is not numeric.")

        preprocess_data_metrics.log_metric("target_dtype_check", "Passed: Numeric")

        # ------------------ 2. Time Series Setup: Date Indexing & Sort ------------------
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df.dropna(subset=[date_col], inplace=True)
            df.sort_values(by=date_col, ascending=True, inplace=True)
            logging.info(f"✅ Set '{date_col}' as DatetimeIndex and sorted data.")
        except Exception as e:
            raise RuntimeError(f"❌ Failed to set '{date_col}' as DatetimeIndex: {e}")

        # ------------------ 3. TARGET VARIABLE HANDLING (Interpolation & Clipping) ------------------

        # ** 3a. Target Variable Null Handling (CRITICAL: Interpolation) **
        try:
            # Try 'time' interpolation first, preferred for DatetimeIndex
            df[target] = df[target].interpolate(method=interpolation_method)
            logging.info(
                f"✅ Interpolated target '{target}' with '{interpolation_method}' method."
            )
        except Exception:
            # Fallback to 'linear' if the specified method fails (e.g., 'time' on non-uniform index)
            df[target] = df[target].interpolate(method="linear")
            logging.info("⚠️ Falling back to 'linear' interpolation for target.")

        # After interpolation, fill remaining NaNs (e.g., at the start/end) with ffill and bfill
        nan_after_interp = df[target].isna().sum()
        if nan_after_interp > 0:
            df[target] = df[target].ffill().bfill()
            logging.warning(
                f"⚠️ Filled {nan_after_interp} remaining NaNs in target with forward-fill and backward-fill."
            )

        # After interpolation, fill remaining NaNs (e.g., at the start/end) with median
        nan_after_interp = df[target].isna().sum()
        if nan_after_interp > 0:
            df[target] = df[target].fillna(df[target].median())
            logging.warning(
                f"⚠️ Filled {nan_after_interp} remaining NaNs in target with median imputation."
            )

        # ** 3b. Target Variable Outlier Handling (Clipping) **
        # Use the parsed tuple
        if clip_q_tuple and clip_q_tuple[0] < clip_q_tuple[1]:
            lower_q, upper_q = df[target].quantile(clip_q_tuple)
            initial_outliers = (df[target] < lower_q).sum() + (
                df[target] > upper_q
            ).sum()
            df[target] = df[target].clip(lower=lower_q, upper=upper_q)
            logging.info(
                f"✅ Clipped target '{target}' (removed {initial_outliers} outliers) to range "
                f"[{clip_q_tuple[0]*100}%, {clip_q_tuple[1]*100}%] quantiles."
            )

        # ------------------ 4. Feature Engineering: Date-Time Features ------------------

        # date_features = [
        #     "year",
        #     "quarter",
        #     "month",
        #     "dayofyear",
        #     "dayofweek",
        #     "is_month_start",
        #     "is_month_end",
        #     "is_quarter_start",
        #     "is_quarter_end",
        #     "is_year_start",
        #     "is_year_end",
        # ]

        # # NOTE: df.index is the DatetimeIndex already. Use it directly.

        # for feature in date_features:
        #     if hasattr(df.index, feature):
        #         # Ensure correct attribute extraction and type casting for booleans
        #         attr = getattr(df.index, feature)
        #         if isinstance(attr, pd.Series) and attr.dtype == bool:
        #             df[feature] = attr.astype(int)
        #         elif isinstance(
        #             attr, bool
        #         ):  # For single index properties like is_month_start, etc.
        #             df[feature] = int(attr)
        #         else:
        #             df[feature] = attr

        # logging.info(f"✅ Created {len(date_features)} date-time features.")

        # Drop the original date column as it's now the index and features are extracted
        # The index is already the date_col. We do NOT drop it or reset it here.
        # df.reset_index(drop=False, inplace=True) # REMOVED: Unnecessary reset/set index

        # ------------------ 5. Feature Selection: Drops (Exogenous Features) ------------------

        current_cols = [col for col in df.columns if col != target and col != date_col]

        # ** 5a. Drop ID-Like Columns **
        if drop_id_like:
            id_cols = [
                col
                for col in current_cols
                if (
                    any(k in col.lower() for k in id_like_keywords)
                    or (df[col].nunique() == len(df))
                )  # Unique values suggests ID
                and col != date_col  # CRITICAL: Never drop date column
            ]
            df.drop(columns=id_cols, errors="ignore", inplace=True)
            logging.info(f"✅ Dropped {len(id_cols)} ID-like columns.")
            current_cols = [
                col for col in df.columns if col != target and col != date_col
            ]

        # ** 5b. Drop High-Null Columns **
        high_null_cols = [
            col
            for col in current_cols
            if df[col].isnull().mean() > (1.0 - min_non_null_ratio)
            and col != date_col  # CRITICAL: Never drop date column
        ]
        df.drop(columns=high_null_cols, errors="ignore", inplace=True)
        logging.info(f"✅ Dropped {len(high_null_cols)} high-null columns.")
        current_cols = [col for col in df.columns if col != target and col != date_col]

        # ------------------ 6. Feature Engineering & Imputation (Exogenous Features) ------------------

        # Re-check types after date feature creation
        numeric_cols = [
            col
            for col in current_cols
            if pd.api.types.is_numeric_dtype(df[col])
            and col != date_col  # Date column should not be treated as numeric feature
        ]
        categorical_cols = [
            col
            for col in current_cols
            if (df[col].dtype == "object" or df[col].dtype.name == "category")
            and (col != date_col)
        ]

        # ** 6a. Numeric Imputation (for remaining NaNs in exogenous features) **
        if numeric_cols:
            num_imputer = SimpleImputer(strategy="median")
            df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

            logging.info(
                f"✅ Imputed NaNs in {len(numeric_cols)} numeric exogenous features with median."
            )

        # ** 6b. Categorical Feature Handling **
        if categorical_cols:
            # Use simpleImputer for consistency (though above fillna is sufficient)
            cat_imputer = SimpleImputer(strategy="most_frequent")
            df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

            # Group rare categories to other
            for col in categorical_cols:
                value_counts = df[col].value_counts(normalize=True)
                rare_categories = value_counts[
                    value_counts < rare_category_threshold
                ].index
                df[col] = np.where(df[col].isin(rare_categories), "other", df[col])

            # Convert to 'category' dtype
            df[col] = df[col].astype("category")

        logging.info(
            f"✅ Handled NaNs and rare categories for {len(categorical_cols)} categorical features."
        )

        # ------------------ 7. Final Feature Selection ------------------

        # ** 7a. Quasi-Constant Feature Removal **
        if numeric_cols:
            # Ensure we only check variance on the numeric columns *still present*
            numeric_final_vt = [col for col in numeric_cols if col in df.columns]
            df_numeric = df[numeric_final_vt].copy()

            # Temporarily fill NaNs for VT (using median is robust)
            df_numeric = df_numeric.fillna(df_numeric.median())

            vt = VarianceThreshold(threshold=(1.0 - quasi_constant_threshold))
            vt.fit(df_numeric)
            low_variance_cols = [
                col for i, col in enumerate(numeric_final_vt) if not vt.get_support()[i]
            ]
            df.drop(columns=low_variance_cols, errors="ignore", inplace=True)
            logging.info(
                f"✅ Dropped {len(low_variance_cols)} quasi-constant numeric features."
            )

        current_cols = [col for col in df.columns if col != target and col != date_col]

        # ** 7b. High Cardinality Categorical Drop **
        # Re-identify final categorical columns
        final_categorical_cols = [
            col
            for col in current_cols
            if (df[col].dtype == "category" or df[col].dtype == "object")
            and (col != date_col)
        ]

        high_card_cols = [
            col
            for col in final_categorical_cols
            if df[col].nunique() > categorical_limit and col in current_cols
        ]
        df.drop(columns=high_card_cols, errors="ignore", inplace=True)
        logging.info(
            f"✅ Dropped {len(high_card_cols)} high-cardinality features (>{categorical_limit} unique values)."
        )

        # ------------------ 8. Correlation Data & Save ------------------

        # Calculate Correlation with Target Variable
        # Calculate the Pearson correlation coefficient between all columns and the target column
        logging.info(
            "✅ Calculated Pearson correlation coefficients with target variable."
        )
        correlation_cols = [
            col
            for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col])
            and col != target
            and col != date_col
        ]
        correlation_series = df[correlation_cols].corrwith(df[target]).abs()
        logging.info(f"✅ Correlation coefficients:\n{correlation_series}")

        # Save correlation data to correlation_data output
        # correlation_series: index = feature, values = correlation
        correlation_df = correlation_series.reset_index()
        correlation_df.columns = ["feature", "correlation"]

        # use abs sum to avoid cancelation; change to raw sum if desired
        correlation_df["percentage_correlation"] = (
            correlation_df["correlation"].abs() * 100.0
        )

        correlation_df.to_csv(correlation_data.path, index=False)

        logging.info(
            f"✅ Saved correlation data to artifact path: {correlation_data.path}"
        )

        # Identify columns where the absolute correlation is less than the threshold
        columns_to_drop = correlation_series[
            correlation_series < correlation_threshold
        ].index.tolist()
        logging.info(
            f"✅ Columns to drop based on correlation threshold ({correlation_threshold}): {columns_to_drop}"
        )
        preprocess_data_metrics.log_metric(
            "correlation_threshold", correlation_threshold
        )
        preprocess_data_metrics.log_metric(
            "low_correlation_columns_dropped", len(columns_to_drop)
        )

        # Log and filter features based on correlation threshold
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            logging.info(
                f"✅ Dropped {len(columns_to_drop)} columns with low correlation to target."
            )
        else:
            logging.info(
                f"✅ No columns found with absolute correlation less than {correlation_threshold}."
            )

        # Final Save - Ensure date and target columns are preserved
        logging.info(
            f"⭐ Saving processed data to artifact path... Dataset Columns: {df.columns.tolist()}"
        )

        # Critical: Ensure date column is in the DataFrame before saving
        if date_col not in df.columns:
            logging.warning(
                f"⚠️ Date column '{date_col}' not in DataFrame. Adding it back."
            )
            # Date column should still be in the original dataframe, don't reset_index

        # Verify both critical columns exist before saving
        if date_col not in df.columns:
            raise RuntimeError(
                f"❌ Critical error: Date column '{date_col}' is missing before save."
            )
        if target not in df.columns:
            raise RuntimeError(
                f"❌ Critical error: Target column '{target}' is missing before save."
            )

        # Save without index to avoid unnamed column
        df.to_csv(preprocess_data.path, index=False)

        rows_final, cols_final = df.shape
        logging.info(
            f"✅ Processed data saved to artifact path. Final shape: {df.shape}"
        )

    # --- Error Handling ---
    except Exception as e:
        logging.error(f"❌ Data preprocessing component failed: {e}")
        raise RuntimeError(f"❌ Data preprocessing component failed: {e}")

    # ------------------ 9. Create Vertex AI Dataset (Optional) ------------------
    try:
        logging.info("⭐ Attempting to create managed Vertex AI Dataset...")
        gcs_uri = preprocess_data.uri
        logging.info(
            f"✅ Initializing Vertex AI: Project='{project_id}', Region='{region}'"
        )
        aiplatform.init(project=project_id, location=region)

        now = pd.Timestamp.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        vertex_dataset_display_name = f"{dataset_display_name}-{timestamp}"

        dataset = aiplatform.TabularDataset.create(
            display_name=vertex_dataset_display_name,
            gcs_source=[gcs_uri],
            sync=True,  # Synchronous creation for component
        )
        logging.info(f"✅ Created Vertex AI Dataset: {dataset.resource_name}")
        preprocess_data_metrics.log_metric(
            "vertex_dataset_name", vertex_dataset_display_name
        )
    except ImportError:
        logging.warning(
            "⚠️ google-cloud-aiplatform not installed. Skipping Vertex AI Dataset creation."
        )
    except exceptions.GoogleAPICallError as e:
        logging.error(f"❌ Failed to create Vertex AI Dataset: {e}")

    # Final Metrics logging
    final_feature_cols = [
        col for col in df.columns if col != target and col != date_col
    ]
    rows_final = df.shape[0]  # Get current row count after drops
    cols_final = df.shape[1]  # Get current col count
    preprocess_data_metrics.log_metric("rows_after", float(rows_final))
    preprocess_data_metrics.log_metric("cols_after", float(cols_final))
    preprocess_data_metrics.log_metric("rows_removed_total", float(rows0 - rows_final))
    preprocess_data_metrics.log_metric(
        "num_features_after",
        float(
            len(
                [
                    col
                    for col in final_feature_cols
                    if pd.api.types.is_numeric_dtype(df[col])
                ]
            )
        ),
    )
    preprocess_data_metrics.log_metric(
        "cat_features_after",
        float(
            len(
                [
                    col
                    for col in final_feature_cols
                    if df[col].dtype.name in ["category", "object"]
                ]
            )
        ),
    )
