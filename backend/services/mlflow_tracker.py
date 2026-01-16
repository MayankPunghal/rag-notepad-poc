"""
MLflow Tracking module for RAG Brain
Logs experiments, metrics, and model artifacts
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from contextlib import contextmanager

# MLflow imports
try:
    import mlflow
    import mlflow.sklearn
    from mlflow.entities import Metric, Param
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from config.settings import settings


# ============================================================================
# MLflow Tracker
# ============================================================================

class MLflowTracker:
    """
    MLflow tracking wrapper for RAG Brain
    Logs experiments, parameters, metrics, and artifacts
    """

    def __init__(self, tracking_uri: str = None, experiment_name: str = None):
        """
        Initialize MLflow tracker

        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of the experiment
        """
        self.tracking_uri = tracking_uri or settings.MLFLOW_TRACKING_URI
        self.experiment_name = experiment_name or settings.MLFLOW_EXPERIMENT_NAME
        self.client = None
        self.experiment_id = None
        self.run_id = None
        self._initialized = False

        if MLFLOW_AVAILABLE:
            self._initialize()

    def _initialize(self):
        """Initialize MLflow connection"""
        if not self.tracking_uri:
            # Use local MLflow directory
            self.tracking_uri = f"file://{settings.MLFLOW_DIR}"
            settings.MLFLOW_DIR.mkdir(parents=True, exist_ok=True)

        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            self.client = MlflowClient(tracking_uri=self.tracking_uri)

            # Get or create experiment
            experiment = self.client.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = self.client.create_experiment(self.experiment_name)
            else:
                experiment_id = experiment.experiment_id

            self.experiment_id = experiment_id
            self._initialized = True
        except Exception as e:
            print(f"Warning: Could not initialize MLflow: {e}")
            self._initialized = False

    @contextmanager
    def start_run(
        self,
        run_name: str = None,
        tags: Dict[str, str] = None
    ):
        """
        Start an MLflow run context

        Args:
            run_name: Name of the run
            tags: Optional tags for the run

        Yields:
            run_id: The ID of the started run
        """
        if not self._initialized:
            yield None
            return

        run_name = run_name or f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        try:
            with mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=run_name,
                tags=tags
            ) as run:
                self.run_id = run.info.run_id
                yield self.run_id
        except Exception as e:
            print(f"Warning: MLflow run failed: {e}")
            yield None

    def log_params(self, params: Dict[str, Any]):
        """Log parameters"""
        if not self._initialized or not self.run_id:
            return

        try:
            for key, value in params.items():
                mlflow.log_param(key, str(value))
        except Exception as e:
            print(f"Warning: Could not log params: {e}")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int = None,
        timestamp: int = None
    ):
        """Log metrics"""
        if not self._initialized or not self.run_id:
            return

        try:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step, timestamp=timestamp)
        except Exception as e:
            print(f"Warning: Could not log metrics: {e}")

    def log_model(
        self,
        model_name: str,
        model: Any,
        artifact_path: str = None
    ):
        """Log a model"""
        if not self._initialized or not self.run_id:
            return

        try:
            artifact_path = artifact_path or model_name
            # For custom models, log as artifact
            mlflow.pyfunc.log_model(artifact_path, python_model=model)
        except Exception as e:
            print(f"Warning: Could not log model: {e}")

    def log_artifact(self, file_path: str, artifact_path: str = None):
        """Log an artifact file"""
        if not self._initialized or not self.run_id:
            return

        try:
            mlflow.log_artifact(file_path, artifact_path)
        except Exception as e:
            print(f"Warning: Could not log artifact: {e}")

    def log_artifacts(self, dir_path: str, artifact_path: str = None):
        """Log all artifacts in a directory"""
        if not self._initialized or not self.run_id:
            return

        try:
            mlflow.log_artifacts(dir_path, artifact_path)
        except Exception as e:
            print(f"Warning: Could not log artifacts: {e}")

    def log_text(self, text: str, filename: str = "output.txt"):
        """Log text as an artifact"""
        if not self._initialized or not self.run_id:
            return

        try:
            mlflow.log_text(text, filename)
        except Exception as e:
            print(f"Warning: Could not log text: {e}")

    def log_dict(self, dictionary: Dict[str, Any], filename: str = "data.json"):
        """Log dictionary as JSON artifact"""
        if not self._initialized or not self.run_id:
            return

        try:
            mlflow.log_dict(dictionary, filename)
        except Exception as e:
            print(f"Warning: Could not log dict: {e}")

    def log_prompt(
        self,
        prompt: str,
        response: str,
        query: str = None,
        context_docs: List[str] = None
    ):
        """Log a prompt-response pair for RAG"""
        if not self._initialized or not self.run_id:
            return

        try:
            prompt_data = {
                "query": query,
                "prompt": prompt,
                "response": response,
                "context_docs": context_docs or [],
                "timestamp": datetime.utcnow().isoformat(),
            }
            mlflow.log_dict(prompt_data, f"prompt_{int(time.time())}.json")
        except Exception as e:
            print(f"Warning: Could not log prompt: {e}")

    def set_tag(self, key: str, value: str):
        """Set a tag on the current run"""
        if not self._initialized or not self.run_id:
            return

        try:
            mlflow.set_tag(key, value)
        except Exception as e:
            print(f"Warning: Could not set tag: {e}")

    def end_run(self, status: str = "FINISHED"):
        """End the current run"""
        if not self._initialized or not self.run_id:
            return

        try:
            mlflow.end_run(status=status)
            self.run_id = None
        except Exception as e:
            print(f"Warning: Could not end run: {e}")

    def get_run_history(self, run_id: str = None) -> Dict[str, Any]:
        """Get run history"""
        if not self._initialized:
            return {}

        try:
            run_id = run_id or self.run_id
            if not run_id:
                return {}

            run = self.client.get_run(run_id)
            return {
                "run_id": run.info.run_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "params": run.data.params,
                "metrics": run.data.metrics,
                "tags": run.data.tags,
            }
        except Exception as e:
            print(f"Warning: Could not get run history: {e}")
            return {}

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments"""
        if not self._initialized:
            return []

        try:
            experiments = self.client.search_experiments()
            return [
                {
                    "id": exp.experiment_id,
                    "name": exp.name,
                    "location": exp.artifact_location,
                }
                for exp in experiments
            ]
        except Exception as e:
            print(f"Warning: Could not list experiments: {e}")
            return []

    def list_runs(self, experiment_id: str = None) -> List[Dict[str, Any]]:
        """List runs in an experiment"""
        if not self._initialized:
            return []

        try:
            experiment_id = experiment_id or self.experiment_id
            runs = self.client.search_runs([experiment_id])
            return [
                {
                    "run_id": run.info.run_id,
                    "name": run.data.tags.get("mlflow.runName", "unnamed"),
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "params": run.data.params,
                    "metrics": run.data.metrics,
                }
                for run in runs
            ]
        except Exception as e:
            print(f"Warning: Could not list runs: {e}")
            return []


# ============================================================================
# RAG-specific MLflow Tracking
# ============================================================================

class RAGMLflowTracker:
    """
    High-level MLflow tracking for RAG operations
    Tracks ingestion, retrieval, and generation metrics
    """

    def __init__(self):
        """Initialize RAG MLflow tracker"""
        self.tracker = MLflowTracker()
        self.ingestion_run = None
        self.retrieval_metrics = []

    def track_ingestion(
        self,
        doc_type: str,
        doc_count: int,
        chunk_count: int,
        total_tokens: int,
        processing_time: float,
        filename: str = None
    ):
        """Track document ingestion metrics"""
        with self.tracker.start_run(
            run_name=f"ingest_{doc_type}_{int(time.time())}",
            tags={"operation": "ingestion", "doc_type": doc_type}
        ) as run_id:
            if run_id:
                self.tracker.log_params({
                    "doc_type": doc_type,
                    "filename": filename or "unknown",
                    "text_model": settings.TEXT_EMBEDDING_MODEL,
                    "chunk_size": settings.CHUNK_SIZE,
                })
                self.tracker.log_metrics({
                    "doc_count": doc_count,
                    "chunk_count": chunk_count,
                    "total_tokens": total_tokens,
                    "processing_time": processing_time,
                    "tokens_per_second": total_tokens / processing_time if processing_time > 0 else 0,
                })

    def track_retrieval(
        self,
        query: str,
        retrieved_count: int,
        avg_score: float,
        retrieval_time: float,
        content_type: str = "all"
    ):
        """Track retrieval metrics"""
        with self.tracker.start_run(
            run_name=f"retrieve_{int(time.time())}",
            tags={"operation": "retrieval", "content_type": content_type}
        ) as run_id:
            if run_id:
                self.tracker.log_params({
                    "content_type": content_type,
                    "top_k": settings.TOP_K_RESULTS,
                })
                self.tracker.log_metrics({
                    "retrieved_count": retrieved_count,
                    "avg_score": avg_score,
                    "retrieval_time": retrieval_time,
                })
                self.tracker.log_text(query, "query.txt")

    def track_generation(
        self,
        query: str,
        context: str,
        response: str,
        context_docs: List[str],
        generation_time: float,
        model: str = "glm-4"
    ):
        """Track RAG generation metrics"""
        with self.tracker.start_run(
            run_name=f"generate_{int(time.time())}",
            tags={"operation": "generation", "model": model}
        ) as run_id:
            if run_id:
                prompt = self.tracker.tracker.build_rag_prompt(
                    query, context, []
                ) if hasattr(self.tracker.tracker, 'build_rag_prompt') else context

                self.tracker.log_params({
                    "model": model,
                    "context_length": len(context),
                    "prompt_length": len(prompt),
                    "context_doc_count": len(context_docs),
                })
                self.tracker.log_metrics({
                    "generation_time": generation_time,
                    "response_length": len(response),
                })
                self.tracker.log_prompt(prompt, response, query, context_docs)

    def get_stats(self) -> Dict[str, Any]:
        """Get tracking statistics"""
        return {
            "mlflow_available": MLFLOW_AVAILABLE,
            "initialized": self.tracker._initialized,
            "tracking_uri": self.tracker.tracking_uri,
            "experiment_name": self.tracker.experiment_name,
        }


# ============================================================================
# Global tracker instance
# ============================================================================

_mlflow_tracker: Optional[RAGMLflowTracker] = None


def get_mlflow_tracker() -> RAGMLflowTracker:
    """Get the global MLflow tracker instance"""
    global _mlflow_tracker
    if _mlflow_tracker is None:
        _mlflow_tracker = RAGMLflowTracker()
    return _mlflow_tracker


def init_mlflow():
    """Initialize MLflow tracking"""
    global _mlflow_tracker
    _mlflow_tracker = RAGMLflowTracker()
    return _mlflow_tracker
