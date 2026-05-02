# Auto-Forecasting MLOps Pipeline 🚀

A comprehensive MLOps pipeline for time series forecasting using Vertex AI, Kubeflow Pipelines, and FastAPI. This project implements end-to-end machine learning workflows for sales and revenue forecasting, and inventory forecasting and notify to client with Prophet, ARIMA/SARIMAX, and Exponential Smoothing models, with inventory consumption projection capabilities.

## 🏗️ Architecture Overview

This repository provides a complete end-to-end time series forecasting solution that:

1. **🪣 Manages GCS bucket lifecycle** for artifact storage
2. **📊 Loads and processes data** from BigQuery (Unicommerce sales data)
3. **🔧 Preprocesses time series data** with feature engineering and date handling
4. **🤖 Trains forecasting models** (Prophet, Auto-ARIMA/SARIMAX, ETS) with automated selection
5. **📈 Evaluates models** with comprehensive time series metrics (RMSE, MAE, MAPE, R²)
6. **🚀 Deploys models** to Vertex AI Model Registry
7. **⚡ Serves forecasts** via FastAPI with inventory consumption projections
8. **� Tracks inventory** by merging forecasts with BigQuery inventory data

## 📁 Repository Structure

```
forecasting-automlops-vertexai-project/
├── 📦 components/                        # Kubeflow Pipeline Components
│   ├── check_bucket.py                    # GCS bucket management
│   ├── data_loading.py                    # Data ingestion from BigQuery
│   ├── data_preprocessing.py              # Time series preprocessing with date/target protection
│   ├── model_trainer.py                   # Model training (Prophet, SARIMAX, ETS)
│   ├── model_evaluation.py                # Model validation & metrics
│   ├── model_deployment.py                # Vertex AI Model Registry deployment
│   └── model_endpoint.py                  # Endpoint creation (optional)
├── 🔄 gitlab-pipelines/                    # CI/CD Configuration
│   ├── __init__.py
│   ├── .gitlab-ci-dev.yml
│   ├── .gitlab-ci-stag.yml
│   └── .gitlab-ci-prod.yml
├── 🧠 pipeline/
│   └── model_pipeline.py                   # Main Kubeflow pipeline definition
├── ⚙️ pipeline_config/
│   └── __init__.py                         # Pipeline configuration class
├── 📜 scripts/
│   ├── run_pipeline.py                     # Pipeline compilation & execution
│   └── forecast_pipeline.py                # Forecasting + inventory logic
├── 🌐 routers/                            # FastAPI Route Modules
│   ├── health.py                           # Health check endpoints
│   ├── training.py                         # Training pipeline triggers
│   └── forecasting.py                      # Forecasting with inventory consumption
├── 🧪 tests/
│   └── test_api.py                         # API test suite
├── ☁️ vertexai-job-pipelines/              # Compiled Pipeline Artifacts
├── 📱 app.py                               # Main FastAPI application
├── 🐳 Dockerfile                           # Container configuration
├── 🔄 .gitlab-ci.yaml                      # GitLab CI/CD pipeline configuration
├── 🔄 .gitignore                           # Specifies intentionally untracked files to ignore
├── 🐳 .dockerignore                        # Specifies files to ignore when building Docker images
├── 📋 requirements.txt                     # Python dependencies
├── 🔑 doreamon-1752016732628-pipeline.json # Service account credentials
└── 📄 template.py                          # Repository templates
```

## 🚀 Quick Start

### Prerequisites

1. **Google Cloud Project** with the following APIs enabled:
   - Vertex AI API
   - Cloud Storage API
   - BigQuery API
   - IAM Service Account Credentials API

2. **Service Account** with appropriate permissions:
   - Storage Admin
   - Vertex AI Admin
   - BigQuery Admin

3. **Python 3.9+** installed

### Setup Instructions

#### 1. Clone and Install Dependencies

```bash
git clone https://github.com/Amirazizgithub/Forecasting-AutoMLOps-VertexAI-Project.git
cd forecasting-automlops-vertexai-project

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. Configure Google Cloud Authentication

```bash
# Place your service account key in the project root
# File should be named: gcp_project_creds.json

# Set environment variable (optional)
export GOOGLE_APPLICATION_CREDENTIALS="gcp_project_creds.json"
```

#### 3. Start the FastAPI Server

```bash
# Start the development server
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Or using Docker
docker build -t auto-forecasting-mlops .
docker run -p 8000:8000 auto-forecasting-mlops

# Server will be available at: http://localhost:8000
# API Documentation: http://localhost:8000/docs
```

#### 4. Run the MLOps Pipeline

```bash
# Compile and run the full pipeline
python scripts/run_pipeline.py

# Or run with custom parameters
python scripts/run_pipeline.py --environment dev
```

## ⚡ FastAPI Endpoints

The application provides the following REST API endpoints:

### 🩺 Health Check
```http
GET /api/v1/health
```
Returns the health status of the forecasting API service.

### 🎯 Forecasting with Inventory Consumption
```http
POST /api/v1/inventory_forecast
Content-Type: application/list[json]

[{
    "CLIENT_NUMBER": 1000000000, 
    "SOURCE_NAME": "unicommerce", 
    "TARGET_VARIABLE": "inventory_demand",
    "ITEMSKU": "AEAD200103",
    "MODEL_PIPELINE": "AEAD200103"
  },
  {
    "forecast_periods":15
  }
]
```

**Response includes:**
- Predicted unit sales for each future date
- Inventory consumption projections
- Historical inventory data merged with forecasts

### 🚀 Training Pipeline
```http
POST /api/v1/train
Content-Type: application/json

{
  "CLIENT_NUMBER": 100000000,
  "SOURCE_NAME": "unicommerce", 
  "TARGET_VARIABLE": "inventory_demand",
  "ITEMSKU": "AEAD200103",
  "MODEL_PIPELINE": "AEAD200103"
}
```

### 📊 API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔧 Configuration & Environment Management

### Project Configuration

Configuration is managed through the `Pipeline_Config` class in `pipeline_config/__init__.py`:

| Parameter | Description | Example |
|-----------|-------------|---------|
| **CLIENT_NUMBER** | Client identifier | `"your_client_number"` |
| **SOURCE_NAME** | Data source name | `"unicommerce"` |
| **TARGET_VARIABLE** | Forecast target | `"unit_sold"` |
| **ITEMSKU** | Product SKU | `"AEBM100101"` |
| **MODEL_PIPELINE** | Pipeline identifier | `"sales_forecast"` |
| **PROJECT_ID** | GCP Project | `"gcp_project_id"` |
| **REGION** | GCP Region | `"us-central1"` |

### Data Sources

**BigQuery Tables:**
- **Sales Data**: `gcp_project_id.Unicommerce.sales_master_data`
  - Contains: date, Unit_Sold, RTO_Rate
  - Aggregated daily sales and returns
- **Inventory Data**: `gcp_project_id.Unicommerce.inventory`
  - Contains: date, itemTypeSKU, inventory
  - Daily inventory levels per SKU

### ML Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Models** | Prophet, SARIMAX, ETS | Time series algorithms |
| **Interpolation** | Time-based | Missing value handling |
| **Date Protection** | Enabled | Date/target never dropped |
| **Model Selection** | RMSE-based | Best model auto-selected |

## 🐳 Docker Support

The project includes full Docker containerization for consistent deployment across environments:

```bash
# Build the Docker image
docker build -t auto-forecasting-mlops .

# Run the container
docker run -p 8000:8000 -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json auto-forecasting-mlops

# Docker Compose (if available)
docker-compose up --build
```

### Docker Configuration
- **Base Image**: `python:3.10-slim`
- **Working Directory**: `/auto-regression-mlops`
- **Exposed Port**: `8000`
- **Health Check**: Built-in via `/api/v1/health` endpoint

## 📊 Pipeline Components

### 1. Bucket Check Component (`check_bucket.py`)
- ✅ Verifies GCS bucket existence
- 🆕 Creates bucket if it doesn't exist
- 🔗 Returns bucket URI for downstream components

### 2. Data Loading Component (`data_loading.py`)
- 📊 Loads sales data from BigQuery (Unicommerce sales_master_data)
- 🏗️ Creates Vertex AI dataset for time series
- 💾 Saves processed data to GCS
- � Validates data quality and schema

### 3. Data Preprocessing Component (`data_preprocessing.py`)
- 🧹 Handles missing values with time-based interpolation
- 🔒 **Critical**: Protects date and target columns from being dropped
- 📏 Handles target variable clipping for outlier reduction
- ✂️ Removes ID-like, high-null, and quasi-constant features
- 📊 Calculates correlation with target variable
- ⚠️ **Key Feature**: Never drops date or target columns during any operation

### 4. Model Training Component (`model_trainer.py`)
- 🤖 Trains multiple time series forecasting models:
  - **Prophet**: Facebook's forecasting model with seasonality
  - **Auto-ARIMA/SARIMAX**: Statistical model with automatic parameter tuning
  - **ETS**: Exponential Smoothing for trend/seasonal patterns
- 🏆 Selects best model based on test RMSE
- 💾 Saves best model artifacts to GCS
- 📈 Handles date column properly for Prophet's ds requirement

### 5. Model Evaluation Component (`model_evaluation.py`)
- 📈 Comprehensive time series model validation
- 📊 Calculates metrics: RMSE, MAE, MAPE, R²
- 🔍 Evaluates forecast accuracy
- 📝 Generates evaluation reports

### 6. Model Deployment Component (`model_deployment.py`)
- 🚀 Deploys model to Vertex AI Model Registry
- 🏷️ Manages model versioning
- 📋 Tracks model metadata (last_train_date, model_type, etc.)
- 🔄 Handles model lifecycle

### 7. Model Endpoint Component (`model_endpoint.py`)
- 🌐 Creates Vertex AI Endpoints for serving (optional)
- ⚖️ Configures auto-scaling
- 🩺 Performs health checks

## 🛠️ Usage Examples

### Pipeline Testing & Validation

```bash
# Test pipeline compilation
python scripts/run_pipeline.py --action compile

# Validate individual components
python -c "from components.check_bucket import check_and_create_gcs_bucket"
python -c "from components.data_loading import load_and_create_dataset_component"

# Run pipeline with test configuration
python scripts/run_pipeline.py --environment dev --client-id test_client_123
```

## � Key Features

### 🔮 Advanced Forecasting Pipeline
- **Multi-Model Support**: Prophet, SARIMAX, and ETS models with automatic selection
- **Intelligent Model Detection**: Properly identifies model types (Prophet vs statsmodels)
- **Date Handling**: Robust datetime conversion and type safety throughout pipeline
- **Inventory Integration**: Merges forecasts with real-time BigQuery inventory data

### 📊 Inventory Consumption Projection
The forecasting pipeline includes unique inventory management features:
1. **Historical Inventory**: Loads actual inventory from BigQuery
2. **Future Projection**: Calculates projected inventory = previous_inventory - predicted_sales
3. **Sequential Calculation**: Day-by-day inventory consumption tracking
4. **Zero-Inventory Handling**: Manages cases with no historical data gracefully

### 🛡️ Data Protection & Quality
- **Column Protection**: Date and target columns never dropped during preprocessing
- **Type Safety**: Proper datetime64[ns] type handling for merges
- **JSON Compliance**: Handles NaN, Infinity values before API responses
- **Int64 Compatibility**: Converts BigQuery Int64 to float for calculations

### 🤖 Model-Specific Handling
**Prophet Models:**
- Uses `make_future_dataframe()` method
- Returns forecast with confidence intervals (yhat, yhat_lower, yhat_upper)
- Requires 'ds' column for dates

**SARIMAX/ARIMA Models:**
- Uses `get_forecast()` method for confidence intervals
- Handles `exog` parameters correctly
- Converts predictions to proper datetime format

**ETS Models:**
- Uses `forecast()` method
- Simple trend and seasonal predictions

## 📈 Model Performance

The pipeline automatically tracks and logs:

- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Square Error  
- **MAE**: Mean Absolute Error
- **Cross-validation scores**
- **Feature importance rankings**

## 🔄 CI/CD Pipeline

### GitLab CI/CD Integration

The project includes comprehensive CI/CD pipelines for all environments:

#### Development Pipeline (`.gitlab-ci-dev.yml`)
- **Triggers**: Push to `develop` branch
- **Stages**: Test → Build → Deploy to Dev
- **Environment**: Development GCP project

#### UAT Pipeline (`.gitlab-ci-uat.yml`)
- **Triggers**: Merge to `stag` branch
- **Stages**: Test → Build → Deploy to UAT
- **Environment**: UAT GCP project

#### Production Pipeline (`.gitlab-ci-prod.yml`)
- **Triggers**: Merge to `product` branch
- **Stages**: Test → Build → Deploy to Production
- **Environment**: Production GCP project

### Pipeline Artifacts

Compiled Kubeflow pipelines are stored in `vertexai-job-pipelines/`:
- Dynamic pipeline generation based on client configuration
- Versioned pipeline definitions
- Environment-specific configurations

## 🛠️ Development Workflow

### Local Development

```bash
# Setup development environment
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Run FastAPI in development mode
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest tests/ -v

# Experiment with Jupyter notebook
jupyter notebook test.ipynb
```

## 🧪 Testing

The project includes a comprehensive test suite using pytest:

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_api.py -v

# Run with coverage
pytest tests/ --cov=app --cov=routers --cov=components -v

# Run tests in Docker
docker run auto-regression-mlops pytest tests/ -v
```

### Test Coverage

| Component | Test Coverage | Description |
|-----------|---------------|-------------|
| **Health Endpoints** | ✅ Complete | Health check functionality |
| **Prediction API** | ✅ Complete | Success, validation, error scenarios |
| **Training API** | ✅ Complete | Pipeline triggers and error handling |
| **CORS/Middleware** | ✅ Complete | Cross-origin and middleware tests |
| **Error Handling** | ✅ Complete | HTTP status codes and exceptions |
| **Integration** | ✅ Complete | End-to-end workflow tests |

### API Testing

```bash
# Test forecasting endpoint with inventory consumption
curl -X POST "http://localhost:8000/api/v1/inventory_forecast" \
  -H "Content-Type: application/list[json]" \
  -d '[{
    "CLIENT_NUMBER": 1000000000, 
    "SOURCE_NAME": "unicommerce", 
    "TARGET_VARIABLE": "inventory_demand",
    "ITEMSKU": "AEAD200103",
    "MODEL_PIPELINE": "AEAD200103"
  },
  {
    "forecast_periods":15
  }
]'

# Test health endpoint
curl http://localhost:8000/api/v1/health

# Test training endpoint
curl -X POST "http://localhost:8000/api/v1/train" \
  -H "Content-Type: application/json" \
  -d '{
  "CLIENT_NUMBER": 100000000, 
  "SOURCE_NAME": "unicommerce", 
  "TARGET_VARIABLE": "inventory_demand",
  "ITEMSKU": "AEAD200103",
  "MODEL_PIPELINE": "AEAD200103"
}'
```

## 📈 Model Performance

The pipeline tracks and logs time series forecasting metrics:
- **RMSE**: Root Mean Square Error (lower is better)
- **MAE**: Mean Absolute Error (average prediction error)
- **MAPE**: Mean Absolute Percentage Error (percentage accuracy)
- **R² Score**: Coefficient of determination (variance explained)
- **Residual Analysis**: Error distribution and autocorrelation
- **Forecast Horizon Accuracy**: Performance across different time periods

### Supported Models Performance
| Model | Use Case | Strengths |
|-------|----------|-----------|
| **Prophet** | Daily/weekly data with seasonality | Handles missing data, holidays, multiple seasonalities |
| **Auto-ARIMA** | Univariate time series | Statistical rigor, confidence intervals |
| **ETS (Exponential Smoothing)** | Simple trend/seasonal patterns | Fast, interpretable, robust baseline |

## 🔄 MLOps Best Practices Implemented

✅ **Automated Model Selection**: Best forecasting model chosen based on RMSE  
✅ **Version Control**: Models versioned in Vertex AI Model Registry  
✅ **Reproducibility**: Fixed package versions, seeded random states  
✅ **Monitoring**: Comprehensive logging and error handling  
✅ **Scalability**: Modular components, containerized execution  
✅ **API-First**: FastAPI with automatic documentation  
✅ **Data Quality**: Time series validation and preprocessing  
✅ **CI/CD Ready**: Structured for automated deployment  
✅ **Exogenous Variables**: Support for external regressors (ad_spends, impressions, etc.)  
✅ **Time Series Best Practices**: Chronological splits, interpolation, seasonality detection

## 🚨 Troubleshooting

### Common Issues & Solutions

#### 1. **Date Column Type Mismatch**
```bash
# Error: "You are trying to merge on object and datetime64[ns] columns"
# Solution: Ensure date columns are datetime64[ns] before merge
# Fixed in forecast_pipeline.py with proper pd.to_datetime() conversions
```

#### 2. **Int64 Type Errors in Calculations**
```bash
# Error: "Invalid value '105995.38' for dtype 'Int64'"
# Solution: Convert BigQuery Int64 columns to float before calculations
# Fixed in final_inventory_consumption_data() method
```

#### 3. **JSON Serialization Errors**
```bash
# Error: "Out of range float values are not JSON compliant"
# Solution: Replace NaN and Infinity values with None
# Fixed in forecasting router with df.replace() and df.fillna()
```

#### 4. **Prophet Model Detection Issue**
```bash
# Error: "'SARIMAXResults' object has no attribute 'make_future_dataframe'"
# Solution: Check for make_future_dataframe() instead of predict()
# Fixed in forecast() method with proper model type detection
```

#### 5. **BigQuery DATE Literal Error**
```bash
# Error: "Invalid DATE literal at [9:35]"
# Solution: Convert Timestamp to string format 'YYYY-MM-DD'
# Fixed with start_date.strftime('%Y-%m-%d')
```

#### 6. **Pipeline Job ID Validation**
```bash
# Error: "pipeline_job_id must be less than 128 characters and only contains [a-z][0-9]-"
# Solution: Convert ITEMSKU to lowercase and replace invalid characters
# Fixed in run_pipeline.py with .lower().replace("_", "-")
```

#### 7. **Preprocessing Date Column Loss**
```bash
# Error: Date column dropped after preprocessing
# Solution: Protect date/target columns throughout preprocessing
# Fixed by adding col != date_col checks in all drop operations
```

#### 8. **Service Account Authentication**
```bash
# Error: Authentication failed
# Solution: Verify service account file exists
ls gcp_project_creds.json
export GOOGLE_APPLICATION_CREDENTIALS="gcp_project_creds.json"
```

#### 9. **Inventory Data Index Error**
```bash
# Error: KeyError: -1 (no historical inventory)
# Solution: Check if nan_start_index > 0 before accessing previous row
# Fixed with conditional check for historical data availability
```

### Debug Mode

```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
python scripts/run_pipeline.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Code Standards

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include type hints
- Write unit tests for new components

## 🆘 Support

For questions and support:

1. Check the [troubleshooting section](#-troubleshooting)
2. Review [Google Cloud documentation](https://cloud.google.com/vertex-ai/docs)
3. Consult [Kubeflow Pipelines documentation](https://www.kubeflow.org/docs/components/pipelines/)
4. Open an issue in this repository

## 🔮 Project Roadmap

### ✅ Completed Features
- [x] Complete MLOps pipeline with 7 Kubeflow components
- [x] FastAPI forecasting service with inventory consumption
- [x] Multi-model support (Prophet, SARIMAX, ETS)
- [x] Vertex AI Model Registry integration
- [x] BigQuery data integration (Unicommerce sales & inventory)
- [x] Robust date/target column protection in preprocessing
- [x] Type-safe datetime handling throughout pipeline
- [x] JSON-compliant API responses (NaN/Infinity handling)
- [x] Intelligent model type detection and forecasting
- [x] Inventory consumption projection with sequential calculation
- [x] Error handling for missing historical data
- [x] Docker containerization
- [x] Comprehensive test suite
- [x] Auto-generated API documentation

### 🚧 In Progress
- [ ] Real-time forecast monitoring dashboard
- [ ] Advanced hyperparameter optimization
- [ ] Model performance tracking over time

### 🔮 Future Enhancements
- [ ] **Multi-SKU Forecasting**: Batch processing for multiple products
- [ ] **Ensemble Models**: Combine Prophet + SARIMAX predictions
- [ ] **Confidence Intervals**: Display uncertainty bounds in API response
- [ ] **Historical Accuracy Tracking**: Compare predictions vs actuals
- [ ] **Automated Retraining**: Trigger retraining when accuracy degrades
- [ ] **Feature Store**: Centralized feature management
- [ ] **A/B Testing**: Champion/challenger model comparison
- [ ] **MLflow Integration**: Experiment tracking and model versioning
- [ ] **Custom Seasonality**: Industry-specific seasonal patterns
- [ ] **Anomaly Detection**: Flag unusual inventory/sales patterns
- [ ] **Multi-Warehouse Support**: Inventory across multiple locations
- [ ] **Promotion Effects**: Handle promotional campaigns in forecasts

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Add tests for new functionality
5. Submit a pull request with detailed description

### Code Standards
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include type hints where possible
- Write unit tests for new components
- Update README for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support & Resources

### Documentation
- **API Docs**: http://localhost:8000/docs (when server is running)
- **Google Cloud Vertex AI**: https://cloud.google.com/vertex-ai/docs
- **Kubeflow Pipelines**: https://www.kubeflow.org/docs/components/pipelines/
- **FastAPI**: https://fastapi.tiangolo.com/

### Getting Help
1. Check the [troubleshooting section](#-troubleshooting)
2. Review the API documentation at `/docs`
3. Examine the generated data profiling reports
4. Open an issue in this repository

### Project Metrics & Statistics
- **📦 Pipeline Components**: 7 (Kubeflow-based for time series forecasting)
- **🌐 API Endpoints**: 3 (Health, Forecast with Inventory, Train)
- **🤖 Forecasting Models**: 3 (Prophet, SARIMAX, ETS)
- **🧪 Test Coverage**: Comprehensive (API, integration, error handling)
- **☁️ Cloud Services**: 4 (Vertex AI, GCS, BigQuery, AI Platform)
- **🐳 Container Support**: Full Docker integration
- **📊 Data Sources**: 2 BigQuery tables (sales_master_data, inventory)
- **🔮 Unique Features**: Inventory consumption projection with sequential tracking
- **🛡️ Data Protection**: Date and target columns never dropped
- **🔄 Type Safety**: Proper datetime64[ns] handling throughout

---

## 🎯 Key Differentiators

| Feature | Status | Implementation |
|---------|--------|----------------|
| **🔄 Inventory Integration** | ✅ Unique | Merges forecasts with BigQuery inventory for consumption projection |
| **�️ Column Protection** | ✅ Production | Date/target columns protected throughout preprocessing |
| **🤖 Smart Model Detection** | ✅ Advanced | Distinguishes Prophet vs SARIMAX by method signature |
| **� Type Safety** | ✅ Robust | Handles datetime64[ns], Int64, and JSON serialization |
| **⚡ Sequential Projection** | ✅ Innovative | Day-by-day inventory calculation: inventory - predicted_sales |
| **🔧 Error Recovery** | ✅ Complete | Graceful handling of missing data, type errors, API issues |

**🚀 Enterprise-Ready Time Series Forecasting with Inventory Management! 🎉**

