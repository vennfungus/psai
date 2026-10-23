"""Sentence-Transformers adapter with lazy model loading."""

from __future__ import annotations

from typing import Any, Sequence


class SentenceTransformerEmbedder:
    """Load a local Sentence-Transformers model only when requested."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Embedding requires sentence-transformers. Install project "
                "dependencies with `python -m pip install -r requirements.txt`."
            ) from exc

        resolved_device = device
        if device == "auto":
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError(
                    "The 'auto' device requires PyTorch. Choose --device cpu "
                    "or install the PyTorch dependency."
                ) from exc
            if torch.cuda.is_available():
                resolved_device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                resolved_device = "mps"
            else:
                resolved_device = "cpu"

        try:
            self._model = SentenceTransformer(
                model_name,
                device=resolved_device,
                local_files_only=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not load embedding model {model_name!r} on "
                f"{resolved_device!r} from local files only. Provision the "
                f"model before running the pipeline: {exc}"
            ) from exc
        self.model_name = model_name
        self.device = resolved_device

    def get_sentence_embedding_dimension(self) -> int:
        """Return the loaded model's fixed output dimension."""
        dimension = self._model.get_sentence_embedding_dimension()
        if not isinstance(dimension, int) or dimension < 1:
            raise RuntimeError(
                "The selected model did not report a valid embedding dimension"
            )
        return dimension

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
    ) -> Any:
        """Encode texts as a NumPy matrix using the selected model."""
        return self._model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=False,
        )
