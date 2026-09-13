"""Mem0 OSS：DeepSeek 提取事实、本地向量模型与 Qdrant 保存/检索。"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from config.project_paths import PROJECT_ROOT
from memory.memory_settings import MEMORY_ENABLED

if TYPE_CHECKING:
    from mem0 import Memory

logger = logging.getLogger("Agent.Memory")
# 本地 Qdrant 由一个进程持有。串行初始化及读写，避免并发请求重复打开存储目录。
_memory_lock = RLock()


def memory_directory() -> Path:
    path = Path(os.getenv("DATA_AGENT_MEMORY_DIR", "runtime/mem0")).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


@lru_cache(maxsize=1)
def _create_client() -> Memory:
    from mem0 import Memory

    root = memory_directory()
    root.mkdir(parents=True, exist_ok=True)
    client = Memory.from_config({
        "llm": {
            "provider": "deepseek",
            "config": {
                "model": os.getenv("DATA_AGENT_MEMORY_LLM_MODEL", "deepseek-v4-flash"),
                "api_key": os.environ["DEEPSEEK_API_KEY"],
                "deepseek_base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                "temperature": 0,
                "top_p": 1,
                "max_tokens": 8192,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": os.getenv("DATA_AGENT_MEMORY_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
                "embedding_dims": 384,
                "model_kwargs": {
                    "device": "cpu",
                    "local_files_only": os.getenv("DATA_AGENT_MEMORY_EMBEDDING_LOCAL_ONLY", "0") == "1",
                },
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "agent_memories",
                "embedding_model_dims": 384,
                "path": str(root / "qdrant"),
                "on_disk": True,
            },
        },
        "history_db_path": str(root / "history.sqlite"),
        "custom_instructions": "保留用户明确表达的事实、偏好、业务约定及适用范围，优先用中文记录。不要把临时要求推广为长期约定，不要把模型推测当作用户事实。",
    })
    client.llm.client = client.llm.client.with_options(timeout=60, max_retries=1)
    logger.info("本地记忆已就绪 backend=mem0-oss directory=%s", root)
    return client


def get_mem0_client() -> Memory:
    if not MEMORY_ENABLED:
        raise RuntimeError("Mem0 is disabled for this process.")
    with _memory_lock:
        return _create_client()


def memory_filters(*, user_id: str, workspace_id: str, thread_id: str) -> dict:
    if not all(value.strip() for value in (user_id, workspace_id, thread_id)):
        raise ValueError("Memory requires user_id, workspace_id and thread_id.")
    # OSS 将自定义 metadata 字段平铺存储；不能照搬托管版的 metadata 嵌套过滤。
    return {"user_id": user_id, "run_id": thread_id, "workspace_id": workspace_id}


def save_memory(content: str, *, user_id: str, workspace_id: str, thread_id: str) -> dict:
    memory_filters(user_id=user_id, workspace_id=workspace_id, thread_id=thread_id)
    with _memory_lock:
        try:
            result = get_mem0_client().add(
                messages=[{"role": "user", "content": content}],
                user_id=user_id,
                run_id=thread_id,
                metadata={"workspace_id": workspace_id, "source_thread_id": thread_id},
            )
        except Exception:
            logger.exception("本地记忆写入失败 thread_id=%s", thread_id)
            return {"status": "FAILED", "message": "记忆保存失败，请查看服务日志。"}
    entries = result["results"]
    logger.info("本地记忆写入完成：%d 条 thread_id=%s", len(entries), thread_id)
    return {
        "status": "SUCCEEDED" if entries else "NO_CHANGE",
        "results": entries,
        "message": "记忆已保存。" if entries else "本次没有新增记忆；未提取到新事实或内容已存在。",
    }


def search_memories(query: str, *, user_id: str, workspace_id: str, thread_id: str) -> list[dict]:
    filters = memory_filters(user_id=user_id, workspace_id=workspace_id, thread_id=thread_id)
    with _memory_lock:
        result = get_mem0_client().search(query=query, filters=filters, top_k=10, threshold=0.1, rerank=False)
    return result["results"]


def _get_scoped_memory(client, memory_id: str, filters: dict) -> dict | None:
    """Mem0 的按 ID 接口不带过滤，写操作前必须检查记录归属。调用方持有锁。"""
    record = client.get(memory_id)
    if not record:
        return None
    metadata = record.get("metadata") or {}
    if (
        record.get("user_id") != filters["user_id"]
        or record.get("run_id") != filters["run_id"]
        or metadata.get("workspace_id") != filters["workspace_id"]
    ):
        return None
    return record


def update_memory_record(
    memory_id: str, content: str, *, user_id: str, workspace_id: str, thread_id: str,
) -> dict:
    filters = memory_filters(user_id=user_id, workspace_id=workspace_id, thread_id=thread_id)
    memory_id, content = memory_id.strip(), content.strip()
    if not memory_id or not content:
        raise ValueError("记忆 ID 和更新内容不能为空。")
    with _memory_lock:
        write_attempted = False
        try:
            client = get_mem0_client()
            record = _get_scoped_memory(client, memory_id, filters)
            if record is None:
                return {"status": "NOT_FOUND", "message": "当前会话中未找到该记忆，未作修改。"}
            if record["memory"] == content:
                return {"status": "NO_CHANGE", "message": "记忆内容相同，无需修改。"}
            # text 是整条替换；SDK 同步更新向量并保留原有身份和 metadata。
            write_attempted = True
            client.update(memory_id=memory_id, text=content)
        except Exception:
            logger.exception("本地记忆修改失败 thread_id=%s memory_id=%s", thread_id, memory_id)
            if write_attempted:
                # SDK 可能先修改向量存储，再在写变更历史时失败，不能假定已回滚。
                return {
                    "status": "FAILED",
                    "invalidate_memory_ids": [memory_id],
                    "message": "记忆修改未完整完成，最终内容需重新检索确认，请查看服务日志。",
                }
            return {"status": "FAILED", "message": "记忆修改失败，请查看服务日志。"}
    logger.info("本地记忆修改完成 thread_id=%s memory_id=%s", thread_id, memory_id)
    return {
        "status": "SUCCEEDED",
        "results": [{"id": memory_id, "event": "UPDATE", "memory": content}],
        "message": "记忆已修改。",
    }


def delete_memory_record(
    memory_id: str, *, user_id: str, workspace_id: str, thread_id: str,
) -> dict:
    filters = memory_filters(user_id=user_id, workspace_id=workspace_id, thread_id=thread_id)
    memory_id = memory_id.strip()
    if not memory_id:
        raise ValueError("记忆 ID 不能为空。")
    with _memory_lock:
        write_attempted = False
        try:
            client = get_mem0_client()
            if _get_scoped_memory(client, memory_id, filters) is None:
                return {"status": "NOT_FOUND", "message": "当前会话中未找到该记忆，未作删除。"}
            write_attempted = True
            client.delete(memory_id=memory_id)
        except Exception:
            logger.exception("本地记忆删除失败 thread_id=%s memory_id=%s", thread_id, memory_id)
            if write_attempted:
                return {
                    "status": "FAILED",
                    "invalidate_memory_ids": [memory_id],
                    "message": "记忆删除未完整完成，最终状态需重新检索确认，请查看服务日志。",
                }
            return {"status": "FAILED", "message": "记忆删除失败，请查看服务日志。"}
    logger.info("本地记忆删除完成 thread_id=%s memory_id=%s", thread_id, memory_id)
    return {
        "status": "SUCCEEDED",
        "results": [{"id": memory_id, "event": "DELETE"}],
        "message": "该条记忆已删除。",
    }


def close_memory_client() -> None:
    with _memory_lock:
        if _create_client.cache_info().currsize:
            _create_client().close()
            _create_client.cache_clear()
