"""Request models shared by the HTTP layer and Agent Runtime."""

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=20000)
    thread_id: str = Field(default="", max_length=128)
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


class ModelSettingsPayload(BaseModel):
    model: Literal["deepseek-v4-pro", "deepseek-v4-flash"]
    api_key: str = Field(default="", max_length=4096)
