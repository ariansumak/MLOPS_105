# pneumoniaclassifier

A short description of the project.

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

## Training the model

During training it is expected that the dataset is mounted at runtime.

The ```data/``` directory must contain the following structure:

```txt 
data/
└── chest_xray/
    ├── train/
    ├── val/
    └── test
```

## Training Locally

Install project as a Python package to register ```pneumoniaclassifier``` as an importable module:

```bash
uv pip install -e .
```

Run the training module in the uv environment (override the CLI paremeters as you like):

```bash
uv run python -m pneumoniaclassifier.train   data.data_dir=data/chest_xray   train.epochs=1   train.device=cpu   wandb.enabled=false
```

## Training with Docker

Build the docker image locally:

```bash
docker build -f dockerfiles/train.dockerfile -t pneumonia-train:latest .
```

Run the docker container (override the CLI paremeters as you like):

```bash
docker run --rm \
  -v $(pwd)/data/chest_xray:/data \
  pneumonia-train \
  data.data_dir=/data \
  train.epochs=1 \
  train.device=cpu \
  wandb.enabled=false
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
