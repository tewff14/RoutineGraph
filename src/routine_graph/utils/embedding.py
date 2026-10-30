from langchain.embeddings import init_embeddings
import numpy as np

class ActivityEmbedder:
    def __init__(self, model: str, provider: str | None = None, **kwargs):
        """Initialize the embedding model, e.g. ActivityEmbedder("openai:text-embedding-3-small")

        model / provider / kwargs are passed to langchain init_embeddings (the provider's package must be installed)
        """
        self.model = init_embeddings(model, provider=provider, **kwargs)

    def embed(self, label: str, description: str) -> np.ndarray:
        """Construct one string from activity label and description, then embed it into a vector"""

        text = f"{label}: {description}"
        return np.array(self.model.embed_query(text))
