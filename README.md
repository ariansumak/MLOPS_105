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

## Training with Docker

Build the docker image locally:

```bash
docker build -f dockerfiles/train.dockerfile -t pneumonia-train:latest .
```

## Run Training

The training container expects the dataset to be mounted at runtime.
The dataset directory must contain the following structure:

```txt 
data/
└── chest_xray/
    ├── train/
    ├── val/
    └── test
```

## Quick Start

```bash
docker run --rm \
  -v $(pwd)/data/chest_xray:/data \
  pneumonia-train \
  data.data_dir=/data
```
This command uses the default training configuration defined in the Hydra config files.

## Override training parameters (recommended)

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
