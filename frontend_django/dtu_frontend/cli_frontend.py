# src/dtu_frontend/cli.py
import os
import sys
import typer

app = typer.Typer()

def _setup_django():
    """Sets up the Django environment."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "settings")
    try:
        import django
        django.setup()
    except ImportError as exc:
        raise ImportError("Django is not installed.") from exc

@app.command()
def dev(port: int = 8000, host: str = "127.0.0.1"):
    """Launch the Django development server."""
    _setup_django()
    from django.core.management import execute_from_command_line
    # We pass the command as if it were typed in manage.py
    execute_from_command_line([sys.argv[0], "runserver", f"{host}:{port}"])

@app.command()
def migrate():
    """Run database migrations."""
    _setup_django()
    from django.core.management import execute_from_command_line
    execute_from_command_line([sys.argv[0], "migrate"])

@app.command()
def setup():
    """Setup command: Migrate and then start the server."""
    _setup_django()
    from django.core.management import execute_from_command_line
    print("🚀 Running migrations...")
    execute_from_command_line([sys.argv[0], "migrate"])
    print("🌐 Starting server...")
    execute_from_command_line([sys.argv[0], "runserver"])

if __name__ == "__main__":
    app()