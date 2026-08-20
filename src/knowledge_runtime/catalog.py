"""Generic loader for directory-based knowledge packages.

This module discovers trusted knowledge cards from a Knowledge Root.
It does not contain database-specific table names or directory names.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import yaml


# 一份 YAML 字典同时包含这些字段时，才把它认作知识卡。
# 这些是“知识卡格式”，不是任何具体数据库的业务规则。
REQUIRED_CARD_FIELDS = frozenset(
    {
        "knowledge_id",
        "knowledge_type",
        "title",
        "summary",
        "payload",
    }
)


@dataclass(frozen=True)
class KnowledgeCard:
    """A knowledge card together with the YAML file that contains it.

    content:
        知识卡本身的字典内容。

    source_path:
        这张知识卡来自哪个 YAML 文件。
        后续发生重复 ID 或格式错误时，可以准确定位来源。
    """

    content: dict
    source_path: Path

    @property
    def knowledge_id(self) -> str:
        """Return the stable identifier of this knowledge card."""

        return self.content["knowledge_id"]

    @property
    def knowledge_type(self) -> str:
        """Return the generic type, such as table, column, or metric."""

        return self.content["knowledge_type"]


def _walk_for_knowledge_cards(node: object) -> Iterator[dict]:
    """Recursively find knowledge-card dictionaries inside a YAML document.

    一个 YAML 文件的顶层结构可能不同：

    第一种可能直接就是一张知识卡：

        knowledge_id: ...
        knowledge_type: ...

    第二种可能把多张卡放在列表里：

        columns:
          - knowledge_id: ...
          - knowledge_id: ...

    第三种可能把卡片包在其他字段下面：

        table_knowledge:
          knowledge_id: ...

    这个函数递归遍历 dict 和 list。只要发现一个字典同时包含
    REQUIRED_CARD_FIELDS，就把它作为一张完整知识卡返回。

    该函数不认识 tables、relationships、metrics 等固定目录，
    也不认识任何具体数据库名称。
    """

    if isinstance(node, dict):
        if REQUIRED_CARD_FIELDS.issubset(node.keys()):
            yield node

            # 已经识别为一张完整知识卡后，不再深入它的 payload。
            # 否则 payload 中碰巧出现类似字段时，可能被重复识别。
            return

        for value in node.values():
            yield from _walk_for_knowledge_cards(value)

    elif isinstance(node, list):
        for item in node:
            yield from _walk_for_knowledge_cards(item)


def load_knowledge_cards(knowledge_root: Path) -> dict[str, KnowledgeCard]:
    """Load every valid YAML knowledge card below a Knowledge Root.

    工作流程：

    1. 检查 Knowledge Root 是否存在。
    2. 递归寻找所有 .yaml 和 .yml 文件。
    3. 解析每份 YAML 文档。
    4. 从文档任意层级发现符合统一格式的知识卡。
    5. 使用 knowledge_id 建立字典索引。
    6. 如果 knowledge_id 重复，立即报错。

    返回格式：

        {
            "table.some_database.some_table": KnowledgeCard(...),
            "column.some_database.some_column": KnowledgeCard(...),
        }

    更换 Knowledge Root 时，只需要传入另一个 Path。
    只要新知识库采用相同的知识卡基础格式，这个函数无需修改。
    """

    knowledge_root = knowledge_root.expanduser().resolve()

    if not knowledge_root.exists():
        raise FileNotFoundError(
            f"Knowledge Root does not exist: {knowledge_root}"
        )

    if not knowledge_root.is_dir():
        raise NotADirectoryError(
            f"Knowledge Root is not a directory: {knowledge_root}"
        )

    yaml_paths = sorted(
        [
            *knowledge_root.rglob("*.yaml"),
            *knowledge_root.rglob("*.yml"),
        ]
    )

    cards: dict[str, KnowledgeCard] = {}

    for path in yaml_paths:
        try:
            with path.open("r", encoding="utf-8") as file:
                document = yaml.safe_load(file)
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid YAML file: {path}") from error

        for content in _walk_for_knowledge_cards(document):
            knowledge_id = str(content["knowledge_id"]).strip()

            if not knowledge_id:
                raise ValueError(
                    f"Empty knowledge_id in file: {path}"
                )

            if knowledge_id in cards:
                previous_path = cards[knowledge_id].source_path
                raise ValueError(
                    "Duplicate knowledge_id "
                    f"{knowledge_id!r}: {previous_path} and {path}"
                )

            cards[knowledge_id] = KnowledgeCard(
                content=content,
                source_path=path,
            )

    if not cards:
        raise ValueError(
            f"No valid knowledge cards found under: {knowledge_root}"
        )

    return cards


def build_catalog(
    cards: dict[str, KnowledgeCard],
) -> dict[str, list[dict]]:
    """Build a minimal directory catalog from loaded knowledge cards.

    这个函数把：

        knowledge_id -> KnowledgeCard

    转换成按照 knowledge_type 分组的目录：

        {
            "table": [...],
            "column": [...],
            "relationship": [...],
            "measure": [...],
            "metric": [...],
        }

    目录名称来自知识卡自身的 knowledge_type，
    不依赖固定文件路径，也不包含具体数据库名称。

    每个目录项只保留：

    - knowledge_id
    - knowledge_type
    - title
    - summary

    不放完整 payload，因为目录浏览阶段只需要帮助 LLM
    判断“应该打开哪一张知识卡”。
    """

    catalog: dict[str, list[dict]] = {}

    for card in cards.values():
        knowledge_type = card.knowledge_type

        item = {
            "knowledge_id": card.knowledge_id,
            "knowledge_type": knowledge_type,
            "title": card.content["title"],
            "summary": card.content["summary"],
        }

        catalog.setdefault(knowledge_type, []).append(item)

    # 保证目录顺序和目录内项目顺序稳定。
    # 相同 Knowledge 每次启动都产生相同顺序，方便测试。
    sorted_catalog: dict[str, list[dict]] = {}

    for knowledge_type in sorted(catalog):
        sorted_catalog[knowledge_type] = sorted(
            catalog[knowledge_type],
            key=lambda item: (
                item["title"].lower(),
                item["knowledge_id"],
            ),
        )

    return sorted_catalog


def read_knowledge_cards(
    cards: dict[str, KnowledgeCard],
    knowledge_ids: list[str],
) -> list[dict]:
    """Return the full content of the requested knowledge cards.

    ``browse_catalog`` and ``search_knowledge_cards`` only expose short
    summaries.  After a caller chooses one or more returned knowledge IDs,
    this function performs the exact ID lookup and returns the full cards.

    The lookup is independent of the YAML file location: moving a card to a
    different directory does not change how it is opened as long as its
    ``knowledge_id`` remains stable.
    """

    missing_ids = [knowledge_id for knowledge_id in knowledge_ids if knowledge_id not in cards]
    if missing_ids:
        raise KeyError(
            "Knowledge IDs do not exist: " + ", ".join(sorted(missing_ids))
        )

    return [cards[knowledge_id].content for knowledge_id in knowledge_ids]


def _searchable_text(card: KnowledgeCard) -> str:
    """Build the small lexical search document for one knowledge card."""

    discovery = card.content.get("discovery", {})
    parts = [
        card.knowledge_id,
        card.knowledge_type,
        card.content.get("title", ""),
        card.content.get("summary", ""),
        *discovery.get("keywords", []),
        *discovery.get("aliases", []),
    ]
    return "\n".join(str(part).lower() for part in parts)


def search_knowledge_cards(
    cards: dict[str, KnowledgeCard],
    query: str,
    knowledge_type: str = "",
) -> list[dict]:
    """Find knowledge cards by lexical metadata matching.

    This is a lightweight shortcut, not RAG: it searches IDs, types, titles,
    summaries, keywords, and aliases.  It returns only directory-style
    summaries and knowledge IDs.  Full content must still be opened through
    ``read_knowledge_cards``.
    """

    normalized_query = query.strip().lower()
    normalized_type = knowledge_type.strip().lower()
    if not normalized_query:
        return []

    query_tokens = normalized_query.replace(",", " ").split()
    matches: list[tuple[int, KnowledgeCard]] = []

    for card in cards.values():
        if normalized_type and card.knowledge_type.lower() != normalized_type:
            continue

        title = str(card.content.get("title", "")).strip().lower()
        knowledge_id = card.knowledge_id.lower()
        summary = str(card.content.get("summary", "")).lower()
        discovery = card.content.get("discovery", {})
        keywords = {
            str(keyword).strip().lower()
            for keyword in discovery.get("keywords", [])
        }
        aliases = {
            str(alias).strip().lower()
            for alias in discovery.get("aliases", [])
        }
        haystack = _searchable_text(card)
        score = 0

        # Exact metadata matches are much stronger than a word merely
        # appearing somewhere in a long summary or keyword list.
        if normalized_query == title:
            score += 100
        if normalized_query in aliases:
            score += 90
        if normalized_query in keywords:
            score += 80
        if normalized_query in title:
            score += 50
        if normalized_query in knowledge_id:
            score += 40
        if normalized_query in summary:
            score += 20

        score += sum(2 for token in query_tokens if token in haystack)

        if score:
            matches.append((score, card))

    matches.sort(
        key=lambda item: (
            -item[0],
            item[1].content["title"].lower(),
            item[1].knowledge_id,
        )
    )

    # 第一版没有分页，只固定返回最相关的前 8 个摘要，避免把大量内容塞给 LLM。
    return [
        {
            "knowledge_id": card.knowledge_id,
            "knowledge_type": card.knowledge_type,
            "title": card.content["title"],
            "summary": card.content["summary"],
        }
        for _, card in matches[:8]
    ]


def browse_catalog(
    catalog: dict[str, list[dict]],
    path: str = "/",
) -> dict:
    """Open one directory in the minimal knowledge catalog.

    支持两种操作：

    1. path="/"
       查看根目录有哪些知识类型。

    2. path="/table"
       查看 table 目录下有哪些知识项。

    这个函数只操作 build_catalog() 生成的内存字典，
    不允许 LLM 直接访问真实文件系统。
    """

    normalized_path = path.strip()

    if not normalized_path:
        normalized_path = "/"

    if not normalized_path.startswith("/"):
        normalized_path = "/" + normalized_path

    # 把 "/table/" 统一处理成 "/table"。
    normalized_path = normalized_path.rstrip("/") or "/"

    # 打开根目录。
    if normalized_path == "/":
        entries = []

        for directory_name, items in catalog.items():
            entries.append(
                {
                    "entry_type": "directory",
                    "name": directory_name,
                    "path": f"/{directory_name}",
                    "item_count": len(items),
                }
            )

        return {
            "path": "/",
            "entries": entries,
        }

    # 当前最简版本只支持一级目录，例如 /table。
    path_parts = [
        part
        for part in normalized_path.split("/")
        if part
    ]

    if len(path_parts) != 1:
        raise KeyError(
            f"Catalog path does not exist: {normalized_path}"
        )

    directory_name = path_parts[0]

    if directory_name not in catalog:
        raise KeyError(
            f"Catalog directory does not exist: {normalized_path}"
        )

    entries = []

    for item in catalog[directory_name]:
        entries.append(
            {
                "entry_type": "knowledge",
                **item,
            }
        )

    return {
        "path": normalized_path,
        "entries": entries,
    }
