from pydantic import BaseModel, Field, model_validator


class NeuGConfig(BaseModel):
    collection_name: str = Field("mem0", description="Name of the collection (mapped to a NeuG node table)")
    embedding_model_dims: int = Field(1536, description="Dimensions of the embedding model")
    db_path: str = Field("/tmp/neug_mem0", description="Path to the NeuG database directory")
    distance: str = Field("cosine", description="Distance metric for HNSW similarity ('cosine' or 'l2')")

    @model_validator(mode="before")
    @classmethod
    def validate_distance(cls, values):
        distance = values.get("distance")
        if distance and distance not in ("cosine", "l2", "euclidean"):
            raise ValueError("Invalid distance for NeuG. Must be one of: 'cosine', 'l2', 'euclidean'")
        return values
