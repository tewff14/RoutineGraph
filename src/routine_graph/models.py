"""Activity validation and the shared temporal representation."""

from datetime import date as CalendarDate, time
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Activity(BaseModel):
    model_config = ConfigDict(extra="allow")

    label: str = Field(min_length=1)
    description: str = Field(min_length=1)
    start_time: str
    end_time: str
    date: CalendarDate | None = None
    subject: str | None = None
    location: str | None = None
    interacted_objects: list[str] = Field(default_factory=list)

    @field_validator("label", "description")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()

    @field_validator("start_time", "end_time")
    @classmethod
    def clock_time(cls, value: str) -> str:
        parsed = time.fromisoformat(value)
        if parsed.tzinfo is not None or parsed.microsecond:
            raise ValueError("use local HH:MM:SS without a timezone or fractional seconds")
        return parsed.isoformat()

    def metadata(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"label", "description", "start_time", "end_time"}, exclude_none=True)


def clock_seconds(value: str) -> int:
    parsed = time.fromisoformat(value)
    return parsed.hour * 3600 + parsed.minute * 60 + parsed.second


def normalized_vectors(vectors: Any, dim: int | None = None) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or (dim is not None and array.shape[1] != dim):
        raise ValueError(f"Expected a matrix of vectors with dimension {dim}; got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Embedding vectors must be finite")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding vectors must be nonzero")
    return np.ascontiguousarray(array / norms)


def temporal_vector(vector: np.ndarray, start_time: str, end_time: str, weight: float = 0.5) -> np.ndarray:
    """Append weighted cyclical start/end coordinates to a unit semantic vector."""
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("Temporal weight must be finite and nonnegative")
    semantic = normalized_vectors([vector])[0]
    angles = np.array([clock_seconds(start_time), clock_seconds(end_time)]) * (2 * np.pi / 86400)
    coordinates = np.column_stack((np.sin(angles), np.cos(angles))).ravel()
    return normalized_vectors([np.concatenate((semantic, weight * coordinates))])[0]
