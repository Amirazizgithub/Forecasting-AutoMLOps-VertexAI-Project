from pathlib import Path

# Root of the repo (use current directory or change as needed)
ROOT = Path(".")  # change to Path("/desired/path/your-repo-name") if needed

# Directories to create
dirs = [
    "components",
    "gitlab-pipelines",
    "pipeline",
    "pipeline_config",
    "scripts",
    "routers",
    "tests",
    "vertexai-job-pipelines",
]

# Files to create (path -> optional initial content)
files = {
    # Components
    "components/check_bucket.py": "",
    "components/data_loading.py": "",
    "components/data_preprocessing.py": "",
    "components/model_trainer.py": "",
    "components/model_evaluation.py": "",
    "components/model_deployment.py": "",
    "components/model_endpoint.py": "",
    # CI/CD
    "gitlab-pipelines/__init__.py": "",
    "gitlab-pipelines/.gitlab-ci-dev.yml": "",
    "gitlab-pipelines/.gitlab-ci-stag.yml": "",
    "gitlab-pipelines/.gitlab-ci-prod.yml": "",
    # Pipeline
    "pipeline/model_pipeline.py": "",
    # Pipeline config
    "pipeline_config/__init__.py": "",
    # Scripts
    "scripts/run_pipeline.py": "",
    "scripts/forecast_pipeline.py": "",
    # Routers
    "routers/health.py": "",
    "routers/training.py": "",
    "routers/forecasting.py": "",
    # Tests
    "tests/test_api.py": "",
    # Root files
    "app.py": "",
    "Dockerfile": "",
    ".gitlab-ci.yaml": "",
    ".gitignore": "",
    ".dockerignore": "",
    "requirements.txt": "",
    "test.ipynb": "",
    "template.py": "",
}


def create_structure(root: Path):
    # Create directories
    for d in dirs:
        (root / d).mkdir(parents=True, exist_ok=True)
    # Create files
    for rel_path, content in files.items():
        p = root / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            # Use .touch for empty files; write content if provided
            if content:
                p.write_text(content, encoding="utf-8")
            else:
                p.touch()


if __name__ == "__main__":
    create_structure(ROOT)
    print("Repository structure created.")
