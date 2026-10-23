from pydantic import BaseModel, Field, field_validator

class AnalyzeRequest(BaseModel):
    object_key: str = Field(min_length=1, max_length=1024)
    evaluation_id: str | None = None
    @field_validator("object_key")
    @classmethod
    def valid_key(cls, value: str) -> str:
        if value.startswith("/") or ".." in value.split("/") or "://" in value:
            raise ValueError("object_key must be a relative R2 object key")
        return value

class StageRequest(BaseModel):
    object_key: str = Field(min_length=1, max_length=1024)
    @field_validator("object_key")
    @classmethod
    def valid_key(cls, value: str) -> str:
        if value.startswith("/") or ".." in value.split("/") or "://" in value:
            raise ValueError("object_key must be a relative R2 object key")
        return value
