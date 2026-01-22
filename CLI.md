To include your new Django frontend CLI, we need to update the **Command Reference** table and add a specific section for **Frontend Development**. This ensures that anyone working on the project knows they can manage both the ML backend and the Web frontend through the same `uv run` interface.

---

# 🛠️ Pneumonia Classifier CLI Guide

This project provides a unified Command Line Interface (CLI) built with **Typer** and **uv** to manage the full machine learning lifecycle and the web frontend.

## 🚀 Getting Started

Before running any commands, ensure your project is installed in **editable mode**. This registers the shortcuts defined in `pyproject.toml`.

```bash
uv pip install -e .

```

---

## 📖 Command Reference

| Category     | Task           | Command                   | Description                                         |
| ------------ | -------------- | ------------------------- | --------------------------------------------------- |
| **Backend**  | **Data Check** | `uv run data-check`       | Verifies data loading and outputs sample counts.    |
| **Backend**  | **Data Stats** | `uv run data-stats`       | Generates distribution plots in `reports/figures/`. |
| **Backend**  | **Training**   | `uv run train`            | Starts the model training pipeline (uses Hydra).    |
| **Backend**  | **Evaluation** | `uv run evaluate`         | Evaluates a checkpoint on the test set.             |
| **Backend**  | **API Server** | `uv run serve-api`        | Starts the FastAPI inference server.                |
| **Frontend** | **Dev Server** | `uv run frontend dev`     | Starts the Django development server.               |
| **Frontend** | **Database**   | `uv run frontend migrate` | Runs Django database migrations.                    |
| **Frontend** | **Full Setup** | `uv run frontend setup`   | Migrates the DB and starts the server in one go.    |

---

## 🖥️ Frontend Management

The frontend is built with Django but managed via the `frontend` CLI wrapper. You no longer need to call `manage.py` directly.

- **Start the Web Interface:**

```bash
uv run frontend dev --port 8001

```

- **Initial Setup (First time running):**
  If you just cloned the repo or updated the database schema, use the setup command:

```bash
uv run frontend setup

```

---

## 🧪 How to Test Your CLI Implementation

### 1. Test the "Help" System

Verify that the frontend subcommands are correctly registered.

```bash
uv run frontend --help

```

### 2. Verify Port Configuration

Test if the Typer argument for the port is working correctly.

```bash
uv run frontend dev --port 8888
# Check if the terminal says: Starting server at http://127.0.0.1:8888/

```

### 3. Verify Integration

Test that you can run the API and the Frontend simultaneously in two different terminals to ensure they don't have port conflicts.

- **Terminal 1:** `uv run serve-api --port 8000`
- **Terminal 2:** `uv run frontend dev --port 8001`

---

## ⚠️ Maintenance Note

If you add new Django management commands or backend scripts, remember to:

1. Update the `app` object in the respective `cli.py` or script file.
2. Ensure the entry point is correctly mapped in `pyproject.toml`.
3. Re-run `uv pip install -e .` to refresh the links.

**Would you like me to help you add a section to this README on how to use `invoke` for multi-step automation (like starting the API and Frontend together)?**
