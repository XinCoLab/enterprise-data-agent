"""Request models shared by the HTTP layer and Agent Runtime."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=20000)
    thread_id: str = Field(default="", max_length=128)
    data_source_id: str | None = Field(default=None, min_length=1, max_length=128)
    knowledge_base_id: str | None = Field(default=None, min_length=1, max_length=128)
    model: Literal["deepseek-v4-pro", "deepseek-v4-flash"] = "deepseek-v4-pro"


class ConversationRenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=80, pattern=r"\S")


class ProfilePayload(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=240)
    backend: Literal["postgresql", "mysql", "duckdb"]
    host: str = Field(default="", max_length=255)
    port: int = Field(default=0, ge=0, le=65535)
    username: str = Field(default="", max_length=255)
    database: str = Field(default="", max_length=255)
    password: str = Field(default="", max_length=2048)
    duckdb_path: str = Field(default="", max_length=2048)
    knowledge_root: str = Field(default="", max_length=2048)


class ProfileReference(BaseModel):
    profile_id: str


class KnowledgeRequest(BaseModel):
    knowledge_root: str = Field(min_length=1, max_length=2048)


class KnowledgeCardCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    knowledge_type: Literal["metric", "glossary_term"]
    title: str = Field(min_length=1, max_length=240, pattern=r"\S")
    summary: str = Field(min_length=1, max_length=20000, pattern=r"\S")
    aliases: list[Annotated[str, Field(max_length=240)]] = Field(default_factory=list, max_length=50)
    definition: str = Field(min_length=1, max_length=40000, pattern=r"\S")
    database_id: str = Field(min_length=1, max_length=255, pattern=r"\S")


class KnowledgeCardPatch(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    title: str = Field(min_length=1, max_length=240, pattern=r"\S")
    summary: str = Field(min_length=1, max_length=20000, pattern=r"\S")
    aliases: list[Annotated[str, Field(max_length=240)]] = Field(max_length=50)
    definition: str = Field(default="", max_length=40000)
    expected_revision: int = Field(ge=1)


class KnowledgeRelationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    source_id: str = Field(min_length=1, max_length=512)
    target_id: str = Field(min_length=1, max_length=512)
    relation: Literal["related_to", "sourced_from", "requires"]
    expected_revision: int = Field(ge=1)


class ModelSettingsPayload(BaseModel):
    model: Literal["deepseek-v4-pro", "deepseek-v4-flash"]
    api_key: str = Field(default="", max_length=4096)
