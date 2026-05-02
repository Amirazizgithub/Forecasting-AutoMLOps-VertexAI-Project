# ==============================================================================
# API ROUTER UNIT TESTS
# ==============================================================================

"""
Api Router Unit Tests
"""

import sys
from fastapi.testclient import TestClient
from unittest.mock import patch
from fastapi.responses import JSONResponse as RealJSONResponse

# Add the current directory to the path to allow imports of modules like 'app'
# In a real setup, you would organize this better, but this makes the test runnable.
sys.path.append(".")

# Assuming the FastAPI app is accessible from a file named 'app.py'
try:
    from app import app
except ImportError:
    # Fallback if 'app' isn't available, but necessary for the tests to function.
    # In a real environment, 'app' would be present.
    raise Exception(
        "Could not import 'app' from 'app.py'. Ensure 'app.py' is accessible."
    )


# Initialize the TestClient for the application
client = TestClient(app)


# ==============================================================================
# 1. Health Router Tests (GET /api/v1/health)
# ==============================================================================


def test_get_health_check_success():
    """Tests a successful health check response."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"message": "Service Health is Good."}


@patch("routers.health.JSONResponse")
def test_get_health_check_api_error(MockJSONResponse):
    """
    Tests the error handling block in the health check endpoint.

    This simulates an error during the successful response generation (the 'try' block),
    forcing the code into the 'except' block to return a 500 error.

    A custom factory function is used as the side_effect to ensure the second call
    (the 500 error response) returns a real JSONResponse object that the TestClient
    can correctly process, preventing the test crash.
    """

    # Use a mutable list to track the call count across calls to the mocked class.
    call_count = [0]

    def mock_json_response_factory(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # 1st call (in try block): Raise the error to force the except path
            raise Exception("Simulated Health Error")

        # 2nd call (in except block): Return a real JSONResponse instance
        # We use the explicitly imported RealJSONResponse class here.
        return RealJSONResponse(*args, **kwargs)

    # Configure the mocked JSONResponse (which is a class) to use the factory function
    MockJSONResponse.side_effect = mock_json_response_factory

    # Make the API call
    response = client.get("/api/v1/health")

    # Assertions:
    # 1. Check that the final status code is 500
    assert response.status_code == 500

    # 2. Check the content
    assert "Simulated Health Error" in response.json()["message"]

    # 3. Verify the mock was called correctly in the error path
    MockJSONResponse.assert_called_with(
        content={"message": "Simulated Health Error"}, status_code=500
    )


# ==============================================================================
# 2. Forecasting Router Tests (POST /api/v1/inventory_forecast)
# ==============================================================================


# Patch the external dependency 'ForecastingPipeline'
@patch("routers.forecasting.ForecastingPipeline")
def test_forecast_pipeline_success(MockForecastingPipeline):
    """Tests a successful forecasting request with inventory consumption."""

    # 1. Setup the mock objects
    mock_pipeline_instance = MockForecastingPipeline.return_value

    # Mock the forecast method to return a DataFrame
    import pandas as pd

    mock_forecast_df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=3),
            "forecast": [100.0, 110.0, 120.0],
        }
    )
    mock_forecast_df = mock_forecast_df.set_index("date")
    mock_pipeline_instance.forecast.return_value = mock_forecast_df

    # Mock the final_inventory_consumption_data method
    mock_final_df = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "predicted_unit_sold": [100, 110, 120],
            "inventory_consumption": [900, 790, 670],
            "total_inventory": [1000, 900, 790],
        }
    )
    mock_pipeline_instance.final_inventory_consumption_data.return_value = mock_final_df

    # 2. Define valid input data
    payload = {
        "CLIENT_ID": "test_client",
        "SOURCE_NAME": "unicommerce",
        "TARGET_VARIABLE": "unit_sold",
        "ITEMSKU": "AEBM100101",
        "MODEL_PIPELINE": "sales_forecast",
        "forecast_periods": 3,
    }

    # 3. Make the API call
    response = client.post("/api/v1/inventory_forecast", json=payload)

    # 4. Assertions
    assert response.status_code == 200
    assert "message" in response.json()
    assert response.json()["message"] == "Forecast successful"
    assert "forecast" in response.json()
    assert len(response.json()["forecast"]) == 3

    # Verify the mock objects were used correctly
    MockForecastingPipeline.assert_called_once()
    mock_pipeline_instance.forecast.assert_called_once()
    mock_pipeline_instance.final_inventory_consumption_data.assert_called_once()


def test_forecast_pipeline_missing_data():
    """Tests the case where config_data or input_data is missing."""

    # Case 1: Empty payload - config_data becomes empty dict with all None values
    # The API checks "if not config_data" which is False for empty dict {}
    # So it proceeds and will fail during pipeline initialization with 500
    response_1 = client.post("/api/v1/inventory_forecast", json={})
    # Actually returns 400 because config_data = {} is falsy when all values are None
    # and input_data = {} is also falsy
    assert response_1.status_code == 400
    assert response_1.json()["message"] == "config_data or input_data is missing"

    # Case 2: List format with empty dicts
    response_2 = client.post("/api/v1/inventory_forecast", json=[{}, {}])
    assert response_2.status_code == 400
    assert response_2.json()["message"] == "config_data or input_data is missing"


@patch("routers.forecasting.ForecastingPipeline")
def test_forecast_pipeline_api_error(MockForecastingPipeline):
    """Tests the general exception handling (API Error)."""

    # 1. Setup the mock to raise an exception during pipeline execution
    MockForecastingPipeline.side_effect = Exception("Model not found in registry")

    # 2. Define valid input data
    payload = {
        "CLIENT_ID": "test_client",
        "SOURCE_NAME": "unicommerce",
        "TARGET_VARIABLE": "unit_sold",
        "ITEMSKU": "AEBM100101",
        "MODEL_PIPELINE": "sales_forecast",
    }

    # 3. Make the API call
    response = client.post("/api/v1/inventory_forecast", json=payload)

    # 4. Assertions
    assert response.status_code == 500
    assert "API Error: Model not found in registry" in response.json()["message"]
    assert "traceback" in response.json()


@patch("routers.forecasting.ForecastingPipeline")
def test_forecast_pipeline_with_list_input(MockForecastingPipeline):
    """Tests forecasting with list input format [config_data, input_data]."""

    # 1. Setup the mock objects
    mock_pipeline_instance = MockForecastingPipeline.return_value

    import pandas as pd

    mock_forecast_df = pd.DataFrame(
        {"date": pd.date_range("2025-01-01", periods=2), "forecast": [100.0, 110.0]}
    )
    mock_forecast_df = mock_forecast_df.set_index("date")
    mock_pipeline_instance.forecast.return_value = mock_forecast_df

    mock_final_df = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02"],
            "predicted_unit_sold": [100, 110],
            "inventory_consumption": [900, 790],
        }
    )
    mock_pipeline_instance.final_inventory_consumption_data.return_value = mock_final_df

    # 2. Define input as list
    payload = [
        {
            "CLIENT_ID": "test_client",
            "SOURCE_NAME": "unicommerce",
            "TARGET_VARIABLE": "unit_sold",
            "ITEMSKU": "AEBM100101",
            "MODEL_PIPELINE": "sales_forecast",
        },
        {"forecast_periods": 2, "exog_features": None},
    ]

    # 3. Make the API call
    response = client.post("/api/v1/inventory_forecast", json=payload)

    # 4. Assertions
    assert response.status_code == 200
    assert response.json()["message"] == "Forecast successful"


# ==============================================================================
# 3. Training Router Tests (POST /api/v1/train)
# ==============================================================================


# Patch the external dependencies
@patch("routers.training.Compile_Run_Pipeline")
@patch("routers.training._forecasting_pipeline_")
def test_compile_run_pipeline_success(MockPipelineFunc, MockCompileRunPipeline):
    """Tests a successful training trigger request."""

    # 1. Setup the mock objects
    mock_runner_instance = MockCompileRunPipeline.return_value
    # Ensure compile_pipeline and run_pipeline methods exist and return None (the default for mocks)
    mock_runner_instance.compile_pipeline.return_value = None
    mock_runner_instance.run_pipeline.return_value = None

    # 2. Define valid input data
    payload = {
        "CLIENT_ID": "test_client",
        "SOURCE_NAME": "unicommerce",
        "TARGET_VARIABLE": "unit_sold",
        "ITEMSKU": "AEBM100101",
        "MODEL_PIPELINE": "sales_forecast",
    }

    # 3. Make the API call
    response = client.post("/api/v1/train", json=payload)

    # 4. Assertions
    assert response.status_code == 200
    assert (
        response.json()["message"]
        == "Training successfully: Vertex AI pipeline's compilation & execution done."
    )

    # Verify the mock objects were used correctly
    MockCompileRunPipeline.assert_called_once_with(data=payload)
    mock_runner_instance.compile_pipeline.assert_called_once_with(
        pipeline_func=MockPipelineFunc
    )
    mock_runner_instance.run_pipeline.assert_called_once()


def test_compile_run_pipeline_missing_data():
    """Tests the case where required training parameters are missing."""

    # Case 1: Missing CLIENT_ID
    payload_1 = {"SOURCE_NAME": "unicommerce", "TARGET_VARIABLE": "unit_sold"}
    response_1 = client.post("/api/v1/train", json=payload_1)
    assert response_1.status_code == 400
    assert (
        response_1.json()["message"]
        == "client_id or source_name or target_variable is missing"
    )

    # Case 2: Missing TARGET_VARIABLE
    payload_2 = {
        "CLIENT_ID": "test_client",
        "SOURCE_NAME": "unicommerce",
    }
    response_2 = client.post("/api/v1/train", json=payload_2)
    assert response_2.status_code == 400
    assert (
        response_2.json()["message"]
        == "client_id or source_name or target_variable is missing"
    )

    # Case 3: Missing SOURCE_NAME
    payload_3 = {"CLIENT_ID": "test_client", "TARGET_VARIABLE": "unit_sold"}
    response_3 = client.post("/api/v1/train", json=payload_3)
    assert response_3.status_code == 400
    assert (
        response_3.json()["message"]
        == "client_id or source_name or target_variable is missing"
    )


@patch("routers.training.Compile_Run_Pipeline")
@patch("routers.training._forecasting_pipeline_")
def test_compile_run_pipeline_api_error(MockPipelineFunc, MockCompileRunPipeline):
    """Tests the general exception handling (API Error) during training."""

    # 1. Setup the mock to raise an exception during pipeline execution
    mock_runner_instance = MockCompileRunPipeline.return_value
    mock_runner_instance.run_pipeline.side_effect = Exception(
        "Vertex AI service is down"
    )

    # 2. Define valid input data
    payload = {
        "CLIENT_ID": "test_client",
        "SOURCE_NAME": "unicommerce",
        "TARGET_VARIABLE": "unit_sold",
        "ITEMSKU": "AEBM100101",
        "MODEL_PIPELINE": "sales_forecast",
    }

    # 3. Make the API call
    response = client.post("/api/v1/train", json=payload)

    # 4. Assertions
    assert response.status_code == 500
    assert response.json()["message"] == "API Error: Vertex AI service is down"


@patch("routers.training.Compile_Run_Pipeline")
@patch("routers.training._forecasting_pipeline_")
def test_compile_run_pipeline_compilation_error(
    MockPipelineFunc, MockCompileRunPipeline
):
    """Tests error handling during pipeline compilation."""

    # 1. Setup the mock to raise an exception during compilation
    mock_runner_instance = MockCompileRunPipeline.return_value
    mock_runner_instance.compile_pipeline.side_effect = Exception(
        "Pipeline compilation failed"
    )

    # 2. Define valid input data
    payload = {
        "CLIENT_ID": "test_client",
        "SOURCE_NAME": "unicommerce",
        "TARGET_VARIABLE": "unit_sold",
    }

    # 3. Make the API call
    response = client.post("/api/v1/train", json=payload)

    # 4. Assertions
    assert response.status_code == 500
    assert "Pipeline compilation failed" in response.json()["message"]
