from invoke import task
import os

@task
def python_check(ctx):
    """Check which Python interpreter is being used."""
    ctx.run("which python" if os.name != "nt" else "where python")

@task
def git(ctx, message):
    """Automate Git add, commit, and push."""
    ctx.run("git add .")
    # Use escaped double quotes \" to satisfy Windows shell requirements
    ctx.run(f'git commit -m "{message}"') 
    ctx.run("git push")

@task
def dvc(ctx, folder="data", message="Add new data"):
    """
    Automate the DVC workflow: add, git commit metadata, and push.
    Usage: uv run invoke dvc --folder 'data/my_folder' --message 'update data'
    """
    ctx.run(f"dvc add {folder}")
    ctx.run(f"git add {folder}.dvc .gitignore")
    ctx.run(f"git commit -m '{message}'")
    ctx.run("git push")
    ctx.run("dvc push")

@task
def pull_data(ctx):
    """Download data from DVC remote."""
    ctx.run("dvc pull")

@task(pre=[pull_data])
def train(ctx, epochs=None):
    """
    Chain task: Pull data first, then run training.
    Usage: uv run invoke train --epochs 5
    """
    cmd = "uv run train"
    if epochs:
        cmd += f" train.epochs={epochs}"
    
    ctx.run(cmd, echo=True)

@task
def dev(ctx):
    """Start both API and Frontend (Requires multiple terminals or backgrounding)."""
    print("🚀 Starting API on port 8000 and Frontend on port 8001...")
    # Note: On Windows, backgrounding is tricky; this is best run in separate terminals.
    ctx.run("start uv run serve-api --port 8000")
    ctx.run("start uv run frontend dev --port 8001")