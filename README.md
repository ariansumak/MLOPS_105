# pneumoniaclassifier

A short description of the project.

## Quickstart: pretrained weights, backend, and frontend

### 1) Download pretrained weights (if training)

The training config uses torchvision pretrained weights. They are downloaded automatically the first time you train.

```bash
uv run python src/pneumoniaclassifier/train.py
```

If you do not want pretrained weights, set `model.pretrained: false` in `configs/train.yaml`.

### 2) Train and save a checkpoint

By default, training saves a checkpoint to `models/m22_model.pt`.

```bash
uv run python src/pneumoniaclassifier/train.py
```

To disable checkpoint saving:

```bash
uv run python src/pneumoniaclassifier/train.py train.save_checkpoint=false
```

### 3) Run the backend (FastAPI)

```bash
uv run uvicorn pneumoniaclassifier.api:app --app-dir src --host 0.0.0.0 --port 8000
```

If the checkpoint path is different, update `configs/inference.yaml` or set `PNEUMONIA_CONFIG`.

### 4) Run the frontend (Django)

```bash
uv run python frontend_django/manage.py runserver 0.0.0.0:8001
```

Open `http://localhost:8001` and point the UI to your backend URL.

### Important to take care of

- Ensure `models/m22_model.pt` exists (or update `configs/inference.yaml`).
- First run may be slower while pretrained weights download.
- Backend and frontend must run on different ports.

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
