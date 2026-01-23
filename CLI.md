# 🛠️ Pneumonia Classifier CLI & Automation Guide

This project features a unified Command Line Interface (CLI) built with **Typer**, **uv**, and **Invoke**. It standardizes how we interact with the ML backend, the Django frontend, and external tools like DVC and Git.

## 🚀 Getting Started

Ensure your environment is synced and the local packages are registered in **editable mode**:

```bash
uv sync
uv pip install -e .

```

---

## 📖 Command Reference

### 1. Python Entry Points (`uv run`)

These are high-level shortcuts defined in `pyproject.toml`. They handle argument validation and provide `--help` menus.

| Category     | Task           | Command                 | Description                                         |
| ------------ | -------------- | ----------------------- | --------------------------------------------------- |
| **Backend**  | **Data Check** | `uv run data-check`     | Verifies data loading and outputs sample counts.    |
| **Backend**  | **Data Stats** | `uv run data-stats`     | Generates distribution plots in `reports/figures/`. |
| **Backend**  | **Training**   | `uv run train`          | Starts model training (Hydra-based).                |
| **Backend**  | **Evaluation** | `uv run evaluate`       | Evaluates a checkpoint on the test set.             |
| **Backend**  | **API Server** | `uv run serve-api`      | Starts the FastAPI inference server.                |
| **Frontend** | **Dev Server** | `uv run frontend dev`   | Starts the Django development server.               |
| **Frontend** | **Full Setup** | `uv run frontend setup` | Migrates DB and starts server in one step.          |

### 2. Automation Tasks (`invoke`)

These are "Task Runner" commands located in `tasks.py`. They automate multi-step workflows involving Git, DVC, and environment setup.

| Task             | Command                                             | Description                                           |
| ---------------- | --------------------------------------------------- | ----------------------------------------------------- |
| **Python Check** | `uv run invoke python-check`                        | Verifies which Python interpreter is being used.      |
| **Git Push**     | `uv run invoke git --message "your message"`        | Automates `add`, `commit`, and `push`.                |
| **DVC Workflow** | `uv run invoke dvc --folder 'data' --message 'msg'` | Adds data to DVC, commits metadata, and pushes.       |
| **Pull Data**    | `uv run invoke pull-data`                           | Downloads data from DVC remote.                       |
| **Full Train**   | `uv run invoke train --epochs 5`                    | **Chained:** Runs `dvc pull` then starts training.    |
| **Dev Servers**  | `uv run invoke dev`                                 | Starts both API (port 8000) and Frontend (port 8001). |

> **Tip:** You can use the alias `uvi` for `uv run invoke` to save keystrokes.
> **Note:** On Windows, the `dev` task is best run in separate terminals due to backgrounding limitations.

---

## ⚙️ Project Structure & Logic

To maintain this CLI, we utilize a **`src` layout**. This ensures code quality by separating the installed package from the project root.

### Configuration Checklist

If you add a new module or package, ensure the following are updated:

1. **`pyproject.toml`**: Update `[tool.setuptools]` to include the new package in `package-dir`.
2. **`__init__.py`**: Every new folder in `src/` must contain this file to be importable.
3. **Entry Points**: Add the function mapping under `[project.scripts]`.

---

## 🧪 Verification & Testing

### 1. The "Help" Check

Verify that Typer is correctly parsing docstrings for the help menu:

```bash
uv run frontend --help

```

### 2. Integration Test (Full Stack)

To verify the system is working end-to-end, open two terminals and run:

- **Terminal 1 (Backend):** `uv run serve-api --port 8000`
- **Terminal 2 (Frontend):** `uv run frontend dev --port 8001`

### 3. Task Chaining Check

Verify that Invoke successfully pulls data before training:

```bash
uv run invoke train
# Observe if 'dvc pull' is logged before training starts.

```

---
