# ==============================================================================
# TIME SERIES FORECASTING MODEL TRAINING TASK
# ==============================================================================

"""
Model Training Component

This component trains machine learning models on the preprocessed data.
"""

# components/model_trainer.py
from kfp.dsl import component, Input, Output, Dataset, Model, Artifact, Metrics


@component(
    packages_to_install=[
        "pandas",
        "numpy==1.24.3",
        "joblib",
        "prophet",
        "pmdarima",
        "statsmodels",
        "scikit-learn",
        "google-api-core",
    ],
    base_image="python:3.10",
)
def model_training_component(
    date_variable: str,
    target_variable: str,
    preprocess_data: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
    feature_importance_artifact: Output[Artifact],
    test_size: float = 0.2,
):
    """
    Trains specialized Time Series forecasting models (Prophet, Auto-ARIMA, ETS)
    on the preprocessed, time-indexed data using a chronological train-test split.
    Selects the best model based on test set RMSE.
    """
    import pandas as pd
    import joblib
    import os
    import json
    import numpy as np
    from itertools import product
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from prophet import Prophet
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.api import ExponentialSmoothing
    import logging
    from google.api_core import exceptions

    # ------------------ Logging Setup ------------------
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info(f"✅ {25*'#'} TIME SERIES FORECASTING TRAINING TASK {25*'#'}")

    # Helper function to calculate metrics
    def calculate_metrics(y_true, y_pred, model_name):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        def mean_absolute_percentage_error(y_true, y_pred):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        mape = mean_absolute_percentage_error(y_true, y_pred)

        # Log and store
        logging.info(f"✅ -> RMSE Test: {rmse:.4f}")
        logging.info(f"✅ -> MAE Test: {mae:.4f}")
        logging.info(f"✅ -> MAPE Test: {mape:.4f}")
        return {
            "rmse_test": float(rmse),
            "mae_test": float(mae),
            "mape_test": float(mape),
            "best_params": {},
        }

    try:
        # 1. Load Data and Time Series Split
        df = pd.read_csv(preprocess_data.path)
        forecast_cols = [date_variable] + [
            col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])
        ]
        df = df[forecast_cols]
        logging.info(f"✅ Forecast columns identified: {forecast_cols}")

        # Assuming the date column is the index or named 'date'
        if date_variable in df.columns:
            last_train_date = df[date_variable].max()
            logging.info(f"✅ Last train date: {last_train_date}")
        elif isinstance(df.index, pd.DatetimeIndex):
            last_train_date = df.index.max()
            logging.info(f"✅ Last train date: {last_train_date}")
        else:
            raise logging.error(f"❌ Error in finding last train date. Error: {str(e)}")

        # Identify exogenous features (all columns except the target)
        exog_cols = [
            col
            for col in df.columns
            if (col != target_variable and col != date_variable)
        ]

        logging.info(
            f"✅ Data Loaded. Exogenous features found: {exog_cols if exog_cols else 'None'}"
        )

        best_models = {}

        # ------------------ 2. Model 1: Facebook Prophet ------------------
        try:
            logging.info(f"\n{'='*60}")
            logging.info(f"🔮 Training Time Series Model: Prophet...")

            # Prepare data for Prophet: must have 'ds' (datestamp) and 'y' (target) columns
            prophet_df = df.copy()
            # Reset index to make date a column instead of index
            prophet_df.reset_index(drop=True, inplace=True)

            categorical_cols = [
                col
                for col in exog_cols
                if pd.api.types.is_string_dtype(prophet_df[col])
            ]
            logging.info(
                f"✅ Categorical exogenous variables for Prophet: {categorical_cols}"
            )

            # One-hot encode categorical exogenous variables for Prophet
            if categorical_cols:
                prophet_df = pd.get_dummies(
                    prophet_df, columns=categorical_cols, drop_first=True
                )
                logging.info(f"✅ One-hot encoded categorical variables for Prophet.")

            if categorical_cols:
                for col in categorical_cols:
                    exog_cols.remove(col)
                    new_cols = [
                        c for c in prophet_df.columns if c.startswith(f"{col}_")
                    ]
                    exog_cols.extend(new_cols)
                logging.info(f"✅ Updated exogenous columns for Prophet: {exog_cols}")

            # Now date_variable should be a column (the index column name from the preprocessed data)
            # Get the actual date column name (it's the index, which becomes a column after reset_index)
            date_col_name = (
                prophet_df.columns[0]
                if prophet_df.index.name is None
                else date_variable
            )
            prophet_df = prophet_df[[date_col_name] + [target_variable] + exog_cols]
            prophet_df.rename(
                columns={date_col_name: "ds", target_variable: "y"}, inplace=True
            )

            # Split again after encoding
            split_point = int(len(prophet_df) * (1 - test_size))

            prophet_df_train = prophet_df.iloc[:split_point]
            prophet_df_test = prophet_df.iloc[split_point:]
            logging.info("✅ Prepared data for Prophet model.")

            # Train Prophet model for validation set
            m_prophet = Prophet(
                seasonality_mode="multiplicative",
                changepoint_prior_scale=0.05,
                yearly_seasonality=False,
                weekly_seasonality=True,
            )

            # Add exogenous regressors (Prophet handles its own feature scaling)
            if exog_cols:
                for col in exog_cols:
                    m_prophet.add_regressor(col, standardize=True)

            m_prophet.fit(prophet_df_train)
            logging.info(f"✅ Prophet model training completed on training data.")

            # Predict
            future_df = prophet_df_test.drop(columns="y")
            forecast = m_prophet.predict(future_df)
            y_test_pred_prophet = forecast["yhat"].values

            # Calculate and store metrics
            prophet_metrics = calculate_metrics(
                prophet_df_test["y"], y_test_pred_prophet, "Prophet"
            )
            prophet_metrics["best_params"] = {
                "exogenous_regressors": exog_cols,
                "seasonality_mode": "multiplicative",
            }

            # Train Prophet model for final model on full data
            final_prophet = Prophet(
                seasonality_mode="multiplicative",
                changepoint_prior_scale=0.05,
                yearly_seasonality=True,
                weekly_seasonality=True,
            )

            # Add exogenous regressors (Prophet handles its own feature scaling)
            if exog_cols:
                for col in exog_cols:
                    final_prophet.add_regressor(col, standardize=True)

            final_prophet.fit(prophet_df)
            logging.info(f"✅ Prophet model training completed on full data.")

            best_models["Prophet"] = {"model_object": final_prophet, **prophet_metrics}
            logging.info(f"✅ Stored Prophet model metrics and parameters.")
        except Exception as e:
            logging.error(f"❌ Prophet model training failed: {e}")
            logging.info("⚠️ Skipping Prophet model and proceeding to next model.")

        # ------------------ 3. Model 2: Auto ARIMA / SARIMAX ------------------
        try:
            logging.info(f"\n{'='*60}")
            logging.info(f"📈 Training Statistical Model: Auto-ARIMA (SARIMAX)...")

            arima_df = df.copy()

            # Prepare exogenous data - IMPORTANT: Handle categorical variables
            exog_cols_sarimax = [
                col
                for col in arima_df.columns
                if (col != target_variable and col != date_variable)
            ]
            logging.info(f"✅ Original exog columns: {exog_cols_sarimax}")

            # One-hot encode categorical variables for SARIMAX
            categorical_cols_sarimax = [
                col
                for col in exog_cols_sarimax
                if pd.api.types.is_string_dtype(arima_df[col])
                or arima_df[col].dtype == "object"
            ]
            logging.info(
                f"✅ Categorical columns to encode: {categorical_cols_sarimax}"
            )

            if categorical_cols_sarimax:
                arima_df = pd.get_dummies(
                    arima_df, columns=categorical_cols_sarimax, drop_first=True
                )
                # Update exog_cols to include encoded columns
                exog_cols_sarimax = [
                    col
                    for col in arima_df.columns
                    if (col != target_variable and col != date_variable)
                ]
                logging.info(f"✅ After encoding, exog columns: {exog_cols_sarimax}")
            else:
                logging.info(f"✅ No categorical columns to encode for SARIMAX.")
                arima_df = arima_df.groupby(date_variable).sum().reset_index()

            # Split data chronologically
            split_point = int(len(arima_df) * (1 - test_size))
            df_train = arima_df.iloc[:split_point]
            df_test = arima_df.iloc[split_point:]
            n_test = len(df_test)

            exog_train = df_train[exog_cols_sarimax] if exog_cols_sarimax else None
            exog_test = df_test[exog_cols_sarimax] if exog_cols_sarimax else None

            # Verify data types
            if exog_train is not None:
                logging.info(f"✅ Exog train dtypes:\n{exog_train.dtypes}")
                # Ensure all columns are numeric
                for col in exog_train.columns:
                    if exog_train[col].dtype == "object":
                        logging.warning(
                            f"⚠️ Column {col} is still object type, converting to numeric"
                        )
                        exog_train[col] = pd.to_numeric(
                            exog_train[col], errors="coerce"
                        )
                        exog_test[col] = pd.to_numeric(exog_test[col], errors="coerce")

            # Define the parameter grids for SARIMAX
            p = d = q = range(0, 3)  # Non-seasonal p, d, q (0, 1, 2)
            P = D = Q = range(0, 2)  # Seasonal P, D, Q (0, 1)
            s7 = 7  # Seasonal period (m=12 for monthly data, m=7 for daily/weekly data)
            s12 = 12  # Seasonal period for monthly data

            # Generate all unique combinations of p, d, q, P, D, Q
            order_pdq = list(product(p, d, q))
            seasonal_pdq = list(product(P, D, Q, [s7])) + list(product(P, D, Q, [s12]))

            # Store the best model and its metric
            best_aic = 10000000
            best_order = None
            best_seasonal_order = None
            best_m_sarimax = None

            logging.info("✅ START MODEL TRAINING")
            # --- Grid Search Loop ---
            for order in order_pdq:
                for seasonal_order in seasonal_pdq:
                    try:
                        m_sarimax = SARIMAX(
                            df_train[target_variable],
                            exog=exog_train,
                            order=order,
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        ).fit(disp=False)

                        # Use AIC for model selection during Grid Search
                        if m_sarimax.aic < best_aic:
                            best_aic = m_sarimax.aic
                            best_order = order
                            best_seasonal_order = seasonal_order
                            best_m_sarimax = m_sarimax
                            logging.info(
                                f"✅ New best model found! AIC: {best_aic:.2f}, Order: {best_order}, Seasonal: {best_seasonal_order}"
                            )
                    except Exception as e:
                        # Skip this combination if it fails
                        print(
                            f"⚠️ Failed for order {order}, seasonal {seasonal_order}: {str(e)[:100]}"
                        )
                        continue

            logging.info("✅ MODEL TRAINING COMPLETE")

            # Predict
            start_index = int(len(df_train))
            end_index = start_index + len(df_test) - 1
            logging.info(f"✅ start_index: {start_index}, end_index: {end_index}")
            y_test_pred_sarimax = best_m_sarimax.predict(
                start=start_index, end=end_index, exog=exog_test
            )

            # Calculate and store metrics
            sarimax_metrics = calculate_metrics(
                df_test[target_variable], y_test_pred_sarimax, "SARIMAX"
            )
            sarimax_metrics["best_params"] = {
                "order": best_order,
                "seasonal_order": best_seasonal_order,
                "exogenous": exog_cols,
            }

            # Train SARIMAX model for final model on full data
            exog_cols_sarimax = [
                col
                for col in arima_df.columns
                if (col != target_variable and col != date_variable)
            ]

            exog_df = arima_df[exog_cols_sarimax] if exog_cols_sarimax else None
            final_sarimax = SARIMAX(
                arima_df[target_variable],
                exog=exog_df,
                order=best_order,
                seasonal_order=best_seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)

            best_models["SARIMAX"] = {"model_object": final_sarimax, **sarimax_metrics}
            logging.info(f"✅ Stored Auto-ARIMA model metrics and parameters.")
        except Exception as e:
            logging.error(f"❌ Auto-ARIMA model training failed: {e}")
            logging.info("⚠️ Skipping Auto-ARIMA model and proceeding to next model.")

        # ------------------ 4. Model 3: Exponential Smoothing (ETS) ------------------

        # logging.info(f"\n{'='*60}")
        # logging.info(f"✨ Training Simplistic Model: Exponential Smoothing (ETS)...")

        # # ETS models are simpler and typically do not handle exogenous features as well,
        # # so we will use a basic multiplicative seasonal model.
        # m_ets = ExponentialSmoothing(
        #     df_train[target_variable],
        #     seasonal_periods=12,
        #     trend="add",
        #     seasonal="mul",
        #     initialization_method="estimated",
        # ).fit()

        # # Predict
        # y_test_pred_ets = m_ets.forecast(steps=n_test)

        # # Calculate and store metrics
        # ets_metrics = calculate_metrics(
        #     df_test[target_variable], y_test_pred_ets, "ETS"
        # )
        # ets_metrics["best_params"] = {
        #     "trend": "add",
        #     "seasonal": "mul",
        #     "seasonal_periods": 12,
        # }
        # best_models["ETS"] = {"model_object": m_ets, **ets_metrics}

        # 5. Select Best Model (Lowest Test RMSE)
        best_model_name = min(best_models.items(), key=lambda x: x[1]["rmse_test"])[0]
        best_model_info = best_models[best_model_name]
        best_model_object = best_model_info["model_object"]

        logging.info(f"\n{'='*60}")
        logging.info(f"🏆 WINNER (Lowest RMSE): {best_model_name}")
        logging.info(f"✅ RMSE Test: {best_model_info['rmse_test']:.4f}")
        logging.info(f"{'='*60}\n")

        # 6. Save Artifacts

        # 6a. Save Best Model
        model_filename = "model.pkl"
        os.makedirs(model.path, exist_ok=True)
        model_artifact_path = os.path.join(model.path, model_filename)
        # Use joblib/pickle for saving
        joblib.dump(best_model_object, model_artifact_path)
        logging.info(
            f"✅ Best model object ({best_model_name}) saved to: {model_artifact_path}"
        )

        # Save metadata
        model.metadata["framework"] = "vertexai-timeseries-forecasting"
        model.metadata["algorithm"] = best_model_name
        model.metadata["last_train_date"] = last_train_date
        model.metadata["mape_test"] = round(float(best_model_info["mape_test"]), 4)
        model.metadata["rmse_test"] = round(float(best_model_info["rmse_test"]), 4)
        model.metadata["mae_test"] = round(float(best_model_info["mae_test"]), 4)
        model.metadata["best_params"] = str(best_model_info["best_params"])
        logging.info(f"✅ Model metadata saved successfully.")

        # Feature Importance Placeholder
        importance_data = {
            "algorithm": best_model_name,
            "note": "Feature importance is not a standard output for statistical forecasting models.",
        }
        importance_filename = "feature_importance.json"
        os.makedirs(feature_importance_artifact.path, exist_ok=True)
        importance_artifact_path = os.path.join(
            feature_importance_artifact.path, importance_filename
        )
        with open(importance_artifact_path, "w") as f:
            json.dump(importance_data, f, indent=4)

        logging.info("✅ Placeholder artifacts saved for pipeline completeness.")

        # 6c. Log Final Metrics
        metrics.log_metric("best_algorithm", best_model_name)
        metrics.log_metric("mape_test", best_model_info["mape_test"])
        metrics.log_metric("rmse_test", best_model_info["rmse_test"])
        metrics.log_metric("mae_test", best_model_info["mae_test"])

        logging.info(f"✅ Final metrics logged successfully.")

    except exceptions.GoogleAPICallError as e:
        logging.error(f"❌ Model training failed: {e}")
        # Reraise the exception to fail the pipeline step
        raise RuntimeError(f"❌ Time Series Model training failed: {e}")
