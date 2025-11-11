import os
from pathlib import Path
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_USER, MLFLOW_SOURCE_NAME, MLFLOW_GIT_COMMIT, MLFLOW_GIT_REPO_URL
import git

def ensure_mlflow_run(experiment_name: str, run_id: str | None = None):
    """Start or resume an MLflow run."""
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment '{experiment_name}' does not exist in MLflow.")
    else:
        exp_id = exp.experiment_id

    # get username from git config (if present),
    # otherwise try to get it from the MLFLOW_TRACKING_USERNAME env var,
    # otherwise fail
    try:
        username = git.Repo(search_parent_directories=True).git.config('--get', 'user.name')
    except Exception:
        username = os.getenv("MLFLOW_TRACKING_USERNAME")
        if username is None:
            raise ValueError("Username not found in git config or MLFLOW_TRACKING_USERNAME environment variable.")

    if run_id is not None:
        run = client.get_run(run_id)
        if run.info.lifecycle_stage == "active":
            print(f"Resuming MLflow run: {run_id}")
        else:
            print(f"Warning: run {run_id} is not active, creating new one.")
            run = client.create_run(experiment_id=exp_id)
    else:
        run = client.create_run(experiment_id=exp_id, tags={
            MLFLOW_SOURCE_NAME: "precompute_clustering_segmentation.py",
            MLFLOW_GIT_REPO_URL: git.Repo(search_parent_directories=True).remotes.origin.url,
            MLFLOW_GIT_COMMIT: git.Repo(search_parent_directories=True).head.commit.hexsha,
            MLFLOW_USER: username,
        })
        print(f"Created new MLflow run: {run.info.run_id}")

    return client, exp_id, run.info.run_id


def artifact_exists(client: MlflowClient, run_id: str, artifact_path: str, file_name: str) -> bool:
    """Check if a file already exists in MLflow artifact store under given run."""
    artifacts = client.list_artifacts(run_id, path=artifact_path)
    for artifact in artifacts:
        if artifact.path.endswith(file_name):
            return True
    return False


def upload_image_if_missing(
    client: MlflowClient,
    run_id: str,
    local_image_path: Path,
    artifact_subdir: str = "result_images"
) -> bool:
    """Upload image to MLflow artifact store if not already there.

    Args:
        client: Active MlflowClient instance.
        run_id: Existing run ID to associate with the upload.
        local_image_path: Path to local image file.
        artifact_subdir: Subdirectory under run artifacts to store the image.
    """
    if not local_image_path.exists():
        raise FileNotFoundError(local_image_path)

    file_name = local_image_path.name
    if artifact_exists(client, run_id, artifact_subdir, file_name):
        print(f"Skipping upload — '{file_name}' already in MLflow under {artifact_subdir}/")
        return False
    else:
        print(f"Uploading '{file_name}' to MLflow under {artifact_subdir}/")
        client.log_artifact(run_id, local_path=local_image_path.as_posix(), artifact_path=artifact_subdir)
        return True


# # ===============================
# # Example Usage
# # ===============================
# if __name__ == "__main__":
#     experiment_name = "image_generation"
#     run_id = None  # You can load your saved run_id here if continuing
#     local_results_dir = Path("results")
#     artifact_subdir = "result_images"

#     # Initialize MLflow run manually
#     client, exp_id, run_id = ensure_mlflow_run(experiment_name, run_id)

#     # Sync local images to MLflow
#     for image_file in sorted(local_results_dir.glob("*.png")):
#         upload_image_if_missing(
#             client=client,
#             run_id=run_id,
#             local_image_path=image_file,
#             artifact_subdir=artifact_subdir
#         )
