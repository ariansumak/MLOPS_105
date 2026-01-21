import os
import typer
import wandb

def link_model(
    artifact_path: str,
    aliases: list[str] = typer.Option(
        ["staging"],
        "-a",
        "--alias",
        help="Aliases to link the artifact with",
        show_default=True,
    ),
):
    """
    Stage a specific model to the W&B model registry.

    Args:
        artifact_path: Path to the artifact to stage.
            Format: "entity/project/artifact_name:version"
        aliases: List of aliases to link the artifact with.

    Example:
        python link_model.py entity/project/artifact_name:version -a staging -a best
    """

    # Validate artifact path
    if not artifact_path:
        typer.echo("ERROR: No artifact path provided. Exiting.")
        raise typer.Exit(code=1)

    # Get W&B environment variables
    entity = os.getenv("WANDB_ENTITY")
    project = "wandb-registry-pneumonia_models"  # You can also make this dynamic if needed

    if not entity:
        typer.echo("ERROR: WANDB_ENTITY is not set in the environment.")
        raise typer.Exit(code=1)

    if not os.getenv("WANDB_API_KEY"):
        typer.echo("ERROR: WANDB_API_KEY is not set in the environment.")
        raise typer.Exit(code=1)

    # Debug prints
    typer.echo(f"Using W&B entity: '{entity}'")
    typer.echo(f"Target project: '{project}'")
    typer.echo(f"Artifact path: '{artifact_path}'")
    typer.echo(f"Aliases: {aliases}")

    # Initialize W&B API
    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": entity, "project": project},
    )

    # Extract artifact name
    try:
        _, _, artifact_name_version = artifact_path.split("/")
        artifact_name, _ = artifact_name_version.split(":")
    except Exception as e:
        typer.echo(f"ERROR: Invalid artifact path format: {artifact_path}")
        raise typer.Exit(code=1) from e

    # Fetch artifact
    artifact = api.artifact(artifact_path)
    typer.echo(f"Artifact name resolved: '{artifact_name}'")

    # Target path for linking in registry
    target_path = f"{entity}/{project}/{artifact_name}"
    typer.echo(f"Linking artifact to: {target_path}")

    # Link artifact
    artifact.link(target_path=target_path, aliases=aliases)
    artifact.save()
    typer.echo(f"Artifact {artifact_path} linked with aliases: {aliases}")


if __name__ == "__main__":
    typer.run(link_model)
