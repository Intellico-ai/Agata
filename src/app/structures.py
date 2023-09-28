from typing import Any, Dict, List, Optional

import numpy as np


class SearchResult:
    """Used to store intermediate results between calls in the gradio app."""

    def __init__(
        self,
        # ids: List[str],
        texts: Optional[List[str]],
        embeddings: Optional[List[np.array]],
        metadatas: Optional[List[Dict[str, Any]]],
    ) -> None:
        # self.ids: List[str] = ids
        self.texts: Optional[List[str]] = texts
        self.embeddings: List[np.array] = embeddings
        self.metadatas: Optional[List[Dict[str, Any]]] = metadatas
        self.additional_infos: Optional[List[Dict[str, Any]]] = None
