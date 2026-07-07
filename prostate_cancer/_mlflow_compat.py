"""Compatibility shim: 3.x MLflow client against a <3.0 tracking server.

A 3.x client's RunsArtifactRepository.list_artifacts() also lists "logged
models" via /api/2.0/mlflow/logged-models/search. A 2.x server doesn't expose
that route and returns 404, which crashes checkpoint logging. The classic
run-artifact checkpoint upload still works against the old server, so we make
the logged-model lookup a no-op when it fails.

Remove this once client and server are on matching major versions.
"""

import logging
from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository


_log = logging.getLogger(__name__)


def apply_mlflow_compat_patch() -> None:
    if getattr(RunsArtifactRepository, "_list_model_artifacts_patched", False):
        return

    _original = RunsArtifactRepository._list_model_artifacts

    def _safe_list_model_artifacts(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return _original(self, *args, **kwargs)
        except MlflowException as exc:
            _log.debug("Skipping logged-model listing (MLflow server < 3.0): %s", exc)
            return []

    RunsArtifactRepository._list_model_artifacts = _safe_list_model_artifacts  # type: ignore[method-assign]
    RunsArtifactRepository._list_model_artifacts_patched = True  # type: ignore[attr-defined]
