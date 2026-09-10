"""Edit narrative Knowledge metadata while retaining the containing YAML document.

The HTTP layer holds RESOURCE_CONFIG_LOCK for the complete read/write/reload
operation. IDs select cards, never filesystem paths. Physical SQL and schema
properties are deliberately outside the edit surface.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import re
from typing import Callable
from uuid import uuid4

import yaml

from knowledge_runtime.catalog import KnowledgeCard, _walk_for_knowledge_cards, load_knowledge_cards
from knowledge_runtime.navigation_graph import build_navigation_graph, explicit_edges


ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+$")
CARD_TYPES = {"database", "table", "column", "relationship", "enum", "measure", "metric", "query_recipe", "warning", "glossary_term"}
DEFINITION_FIELDS = {"metric": "business_definition", "glossary_term": "definition"}
RELATION_FIELDS = {
    "related_to": ("discovery", "related_knowledge_ids"),
    "sourced_from": ("payload", "source_knowledge_ids"),
    "requires": ("payload", "required_knowledge_ids"),
}


class KnowledgeEditError(ValueError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def _valid_id(knowledge_id: str) -> None:
    if not isinstance(knowledge_id, str) or len(knowledge_id) > 512 or not ID_PATTERN.fullmatch(knowledge_id):
        raise KnowledgeEditError("知识 ID 格式不合法。")


def _confined_path(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if resolved == root or root not in resolved.parents:
        raise KnowledgeEditError("知识文件路径超出当前知识库。")
    return resolved


def _revision(content: dict) -> int:
    value = content.get("revision", 1)
    if type(value) is not int or value < 1:
        raise KnowledgeEditError("知识卡 revision 必须为正整数。")
    return value


def _validate_card(card: KnowledgeCard) -> None:
    content = card.content
    _valid_id(card.knowledge_id)
    if card.knowledge_type not in CARD_TYPES:
        raise KnowledgeEditError("知识卡类型不受支持。")
    for field in ("title", "summary"):
        if not isinstance(content.get(field), str) or not content[field].strip():
            raise KnowledgeEditError("知识卡标题和摘要不能为空。")
    if not isinstance(content.get("payload"), dict) or not isinstance(content.get("discovery", {}), dict):
        raise KnowledgeEditError("知识卡 payload 和 discovery 必须为对象。")
    database_id = content.get("database_id", "")
    if not isinstance(database_id, str):
        raise KnowledgeEditError("知识卡 database_id 必须为字符串。")
    discovery = content.get("discovery", {})
    for field in ("aliases", "keywords"):
        values = discovery.get(field, [])
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise KnowledgeEditError("知识卡别名和关键词必须为字符串列表。")
    evidence = content.get("evidence_refs", [])
    if not isinstance(evidence, list) or any(not isinstance(item, dict) for item in evidence):
        raise KnowledgeEditError("知识卡证据引用必须为对象列表。")
    definition_key = DEFINITION_FIELDS.get(card.knowledge_type)
    if definition_key and not isinstance(content["payload"].get(definition_key, ""), (str, type(None))):
        raise KnowledgeEditError("知识卡定义必须为文本。")
    id_lists = [*RELATION_FIELDS.values(), ("payload", "table_ids"), ("payload", "applies_to")]
    for section, field in id_lists:
        values = content.get(section, {}).get(field, [])
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise KnowledgeEditError("知识关联必须为 ID 列表。")
        if len(set(values)) != len(values):
            raise KnowledgeEditError("知识卡包含重复关联。")
        for value in values:
            _valid_id(value)
    payload = content["payload"]
    for field in ("table_id", "column_id", "default_time_field_id"):
        if payload.get(field) is not None:
            _valid_id(payload[field])
    for field in ("from", "to"):
        if field in payload:
            if not isinstance(payload[field], dict):
                raise KnowledgeEditError("知识关系端点必须为对象。")
            if "table_id" in payload[field]:
                _valid_id(payload[field]["table_id"])
    _revision(content)


class KnowledgeEditor:
    """One locked operation over the active root and its real card metadata."""

    def __init__(self, root: Path, reload_runtime: Callable[[Path], None]):
        self.root = root.expanduser().resolve()
        self.reload_runtime = reload_runtime
        # Check symlinks before the generic loader reads anything outside root.
        if not self.root.is_dir():
            raise KnowledgeEditError("当前知识库目录不存在。", 409)
        for path in [*self.root.rglob("*.yaml"), *self.root.rglob("*.yml")]:
            _confined_path(self.root, path)
        try:
            self.cards = load_knowledge_cards(self.root)
            self._validate_bundle(self.cards)
        except KnowledgeEditError:
            raise
        except (ValueError, OSError, TypeError, KeyError) as error:
            raise KnowledgeEditError("当前知识库格式或关联无效，请检查 YAML 文件。") from error

    @staticmethod
    def _validate_bundle(cards: dict[str, KnowledgeCard]) -> None:
        for card in cards.values():
            _validate_card(card)
            if any(edge.source_id == edge.target_id for edge in explicit_edges(card)):
                raise KnowledgeEditError("知识卡不能关联自身。")
        try:
            build_navigation_graph(cards)
        except ValueError as error:
            raise KnowledgeEditError("知识关联指向不存在的卡片。") from error
        target_types = {"contains": "table", "belongs_to": "table", "describes": "column",
                        "default_time_field": "column", "from_table": "table", "to_table": "table"}
        for card in cards.values():
            for edge in explicit_edges(card):
                required_type = target_types.get(edge.relation)
                if required_type and cards[edge.target_id].knowledge_type != required_type:
                    raise KnowledgeEditError("知识关联目标类型不匹配。")

    def _card(self, knowledge_id: str) -> KnowledgeCard:
        _valid_id(knowledge_id)
        if knowledge_id not in self.cards:
            raise KnowledgeEditError("知识卡不存在。", 404)
        return self.cards[knowledge_id]

    def read(self, knowledge_id: str) -> dict:
        card = self._card(knowledge_id)
        content = card.content
        definition_key = DEFINITION_FIELDS.get(card.knowledge_type)
        relations = sorted({edge for item in self.cards.values() for edge in explicit_edges(item)
                            if knowledge_id in (edge.source_id, edge.target_id)})
        return {
            "knowledge_id": card.knowledge_id,
            "knowledge_type": card.knowledge_type,
            "title": content["title"],
            "summary": content["summary"],
            "database_id": content.get("database_id", ""),
            "revision": _revision(content),
            "status": content.get("status", "DRAFT"),
            "aliases": list(content.get("discovery", {}).get("aliases", [])),
            "definition": (content["payload"].get(definition_key) or "") if definition_key else "",
            "source_file": _confined_path(self.root, card.source_path).relative_to(self.root).as_posix(),
            "evidence_refs": deepcopy(content.get("evidence_refs", [])),
            "payload": deepcopy(content["payload"]),
            "relations": [{"source_id": edge.source_id, "target_id": edge.target_id,
                           "relation": edge.relation,
                           "source_title": self.cards[edge.source_id].content["title"],
                           "target_title": self.cards[edge.target_id].content["title"]} for edge in relations],
        }

    def _check_revision(self, card: KnowledgeCard, expected_revision: int) -> None:
        if type(expected_revision) is not int or expected_revision != _revision(card.content):
            raise KnowledgeEditError("知识卡已被更新，请重新加载后再保存。", 409)

    def _save(self, knowledge_id: str, content: dict, path: Path, *, create: bool = False) -> dict:
        path = _confined_path(self.root, path)
        original = None if create else path.read_bytes()
        if create:
            if path.exists() or knowledge_id in self.cards:
                raise KnowledgeEditError("知识 ID 已存在，请重试。", 409)
            document = content
        else:
            document = yaml.safe_load(original.decode("utf-8-sig"))
            matches = [item for item in _walk_for_knowledge_cards(document) if item["knowledge_id"] == knowledge_id]
            if len(matches) != 1:
                raise KnowledgeEditError("知识文件已改变，请重新加载。", 409)
            if _revision(matches[0]) != _revision(self.cards[knowledge_id].content):
                raise KnowledgeEditError("知识文件已改变，请重新加载。", 409)
            matches[0].clear()
            matches[0].update(content)

        candidate = dict(self.cards)
        candidate[knowledge_id] = KnowledgeCard(content, path)
        self._validate_bundle(candidate)
        serialized = yaml.safe_dump(document, allow_unicode=True, sort_keys=False).encode("utf-8")
        temporary = _confined_path(self.root, path.with_name(f".{path.name}.{uuid4().hex}.tmp"))
        try:
            with temporary.open("xb") as stream:
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            try:
                # The runtime builds its entire bundle before replacing globals.
                self.reload_runtime(self.root)
            except Exception as error:
                if original is None:
                    path.unlink()
                else:
                    with temporary.open("xb") as stream:
                        stream.write(original)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, path)
                raise KnowledgeEditError("知识库重新加载失败，文件已恢复。", 500) from error
        finally:
            temporary.unlink(missing_ok=True)
        self.cards = candidate
        return self.read(knowledge_id)

    def patch(self, knowledge_id: str, changes: dict, expected_revision: int) -> dict:
        card = self._card(knowledge_id)
        self._check_revision(card, expected_revision)
        if set(changes) - {"title", "summary", "aliases", "definition"}:
            raise KnowledgeEditError("只能编辑标题、摘要、别名和语义定义。")
        content = deepcopy(card.content)
        for field in ("title", "summary"):
            if field in changes:
                content[field] = changes[field].strip()
        if "aliases" in changes:
            content.setdefault("discovery", {})["aliases"] = list(dict.fromkeys(alias.strip() for alias in changes["aliases"] if alias.strip()))
        if "definition" in changes:
            definition_key = DEFINITION_FIELDS.get(card.knowledge_type)
            definition = changes["definition"].strip()
            if not definition_key:
                if definition:
                    raise KnowledgeEditError("只有指标和业务术语支持编辑定义。")
            else:
                if not definition:
                    raise KnowledgeEditError("语义定义不能为空。")
                if definition != (content["payload"].get(definition_key) or ""):
                    content["status"] = "DRAFT"
                content["payload"][definition_key] = definition
        content["revision"] = _revision(card.content) + 1
        content["updated_at"] = datetime.now(timezone.utc).isoformat()
        return self._save(knowledge_id, content, card.source_path)

    def create(self, values: dict) -> dict:
        knowledge_type = values["knowledge_type"]
        if knowledge_type not in DEFINITION_FIELDS:
            raise KnowledgeEditError("仅支持新建指标和业务术语。")
        database_id = values["database_id"].strip()
        known_databases = {card.content.get("database_id") for card in self.cards.values()}
        if not database_id or database_id not in known_databases:
            raise KnowledgeEditError("请选择当前知识库中已有的数据源标识。")
        knowledge_id = f"{knowledge_type}.manual.{uuid4().hex}"
        title, summary, definition = (values[key].strip() for key in ("title", "summary", "definition"))
        if not title or not summary or not definition:
            raise KnowledgeEditError("标题、摘要和语义定义不能为空。")
        if knowledge_type == "metric":
            payload = {"business_definition": definition, "grain": None,
                       "formula_sql": "UNRESOLVED: 此草稿尚未验证 SQL 公式与物理字段映射。",
                       "source_knowledge_ids": [], "default_time_field_id": None,
                       "rounding_rule": None, "usage_warnings": ["DRAFT：业务定义尚未经过验证。"]}
        else:
            payload = {"term": title, "definition": definition, "excluded_meanings": []}
        content = {"knowledge_id": knowledge_id, "knowledge_type": knowledge_type,
                   "database_id": database_id, "title": title, "summary": summary, "payload": payload,
                   "discovery": {"keywords": [], "aliases": list(dict.fromkeys(alias.strip() for alias in values.get("aliases", []) if alias.strip())), "related_knowledge_ids": []},
                   "evidence_refs": [], "status": "DRAFT", "revision": 1,
                   "updated_at": datetime.now(timezone.utc).isoformat()}
        return self._save(knowledge_id, content, self.root / f"{knowledge_id}.yaml", create=True)

    def add_relation(self, source_id: str, target_id: str, relation: str, expected_revision: int) -> dict:
        source, target = self._card(source_id), self._card(target_id)
        self._check_revision(source, expected_revision)
        if relation not in RELATION_FIELDS:
            raise KnowledgeEditError("关系类型不受支持。")
        if relation == "sourced_from" and source.knowledge_type != "metric":
            raise KnowledgeEditError("只有指标支持添加来源关系。")
        if relation == "requires" and source.knowledge_type != "query_recipe":
            raise KnowledgeEditError("只有查询方案支持添加依赖关系。")
        if source_id == target_id:
            raise KnowledgeEditError("知识卡不能关联自身。")
        if source.content.get("database_id") != target.content.get("database_id"):
            raise KnowledgeEditError("只能关联同一数据源的知识卡。")
        if any(edge.relation == relation and edge.target_id == target_id for edge in explicit_edges(source)):
            raise KnowledgeEditError("该关联已经存在。", 409)
        content = deepcopy(source.content)
        section, field = RELATION_FIELDS[relation]
        content.setdefault(section, {}).setdefault(field, []).append(target_id)
        if relation in {"sourced_from", "requires"}:
            content["status"] = "DRAFT"
        content["revision"] = _revision(source.content) + 1
        content["updated_at"] = datetime.now(timezone.utc).isoformat()
        return self._save(source_id, content, source.source_path)
