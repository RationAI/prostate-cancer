from mlflow import MlflowClient
from rationai.mlkit.lightning.loggers.mlflow import MLFlowLogger


ERR_LOG_MSG = "Logging checkpoint title failed: {}"


def log_checkpoint_title(logger: MLFlowLogger, checkpoint: str) -> None:
    """Log the checkpoint title to the current MLflow run.

    Arguments:
        logger (MLFlowLogger): MLflow logger instance.
        checkpoint (str): MLflow model URI in format
            `mlflow-artifacts:/<experiment_id>/<run_id>/...`.
    """
    # Extract run_id from the checkpoint URI
    run_id = checkpoint.split("/")[2]
    client = MlflowClient()

    # Get the source run's tags or params
    source_run = client.get_run(run_id)

    # Get the run name from tags
    title = source_run.data.tags.get("mlflow.runName")
    if not title:
        print(ERR_LOG_MSG.format("No run name found in tags."))
        return

    # Log it as a parameter to the current run
    current_run_id = logger.run_id

    if current_run_id is None:
        print(ERR_LOG_MSG.format("No current run ID found."))
        return

    client.log_param(current_run_id, "checkpoint_title", title)
