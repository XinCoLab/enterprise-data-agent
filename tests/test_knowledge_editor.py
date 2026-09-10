"""Exercise real YAML editing and runtime reload against isolated roots only."""

from copy import deepcopy
from dataclasses import replace

import pytest
import yaml

from knowledge_runtime.catalog import load_knowledge_cards
from knowledge_runtime.editor import KnowledgeEditError, KnowledgeEditor


def card(kind, name, *, database="example", payload=None):
    return {
        "knowledge_id": f"{kind}.{database}.{name}",
        "knowledge_type": kind,
        "database_id": database,
        "title": name,
        "summary": f"{name} 原始摘要",
        "payload": payload or {},
        "discovery": {"keywords": [name], "aliases": [name], "related_knowledge_ids": []},
        "evidence_refs": [{"evidence_id": f"evidence.{name}", "locator": "原始文档"}],
        "status": "VALIDATED",
        "revision": 1,
        "custom_metadata": {"preserve": [1, 2, 3]},
    }


@pytest.fixture
def knowledge_workspace(tmp_path, monkeypatch):
    from api import configuration_app
    from knowledge_runtime import current_knowledge

    root = tmp_path / "knowledge"
    root.mkdir()
    document = {
        "package_metadata": {"version": "test", "keep": True},
        "table_knowledge": card("table", "orders", payload={"physical_name": "orders"}),
        "nested": {"cards": [
            card("metric", "revenue", payload={"business_definition": "原始收入定义", "formula_sql": "SUM(amount)", "source_knowledge_ids": []}),
            card("glossary_term", "customer", payload={"term": "customer", "definition": "原始客户定义", "excluded_meanings": []}),
            card("column", "amount", payload={"physical_name": "amount", "physical_type": "numeric", "table_id": "table.example.orders"}),
            card("query_recipe", "report", payload={"required_knowledge_ids": [], "reference_sql": "SELECT 1"}),
            card("table", "foreign", database="other"),
        ]},
    }
    source = root / "cards.yaml"
    source.write_text(yaml.safe_dump(document, allow_unicode=True, sort_keys=False), encoding="utf-8")
    for field in ("KNOWLEDGE_CARDS", "KNOWLEDGE_CATALOG", "KNOWLEDGE_NAVIGATION_GRAPH", "KNOWLEDGE_NAVIGATION_GRAPH_TEXT", "_ACTIVE_KNOWLEDGE_SUMMARY"):
        monkeypatch.setattr(current_knowledge, field, getattr(current_knowledge, field))
    current_knowledge.reload_knowledge(root)
    monkeypatch.setattr(configuration_app, "_active_payload", lambda: {"knowledge_root": str(root)})
    monkeypatch.setattr(configuration_app, "has_active_runs", lambda: False)
    return root, source, document


def edit_body(**changes):
    return {"title": "修改后的标题", "summary": "修改后的摘要", "aliases": ["新别名"], "expected_revision": 1, **changes}


def create_body(kind="metric", **changes):
    return {"knowledge_type": kind, "title": "新增知识", "summary": "手工维护摘要", "aliases": ["别名"], "definition": "手工业务定义", "database_id": "example", **changes}


def test_get_patch_preserves_siblings_metadata_and_reload(client, knowledge_workspace):
    from knowledge_runtime import current_knowledge

    root, source, original = knowledge_workspace
    knowledge_id = "metric.example.revenue"
    detail = client.get(f"/api/knowledge-cards/{knowledge_id}")
    assert detail.status_code == 200
    assert detail.json()["card"]["definition"] == "原始收入定义"
    assert detail.json()["card"]["source_file"] == "cards.yaml"
    assert detail.json()["card"]["database_id"] == "example"
    graph = client.get("/api/knowledge-graph").json()
    assert graph["database_ids"] == ["example", "other"]
    metric_node = next(node for node in graph["nodes"] if node["knowledge_id"] == knowledge_id)
    assert (metric_node["database_id"], metric_node["revision"], metric_node["status"]) == ("example", 1, "VALIDATED")
    response = client.patch(f"/api/knowledge-cards/{knowledge_id}", json=edit_body(definition="新收入定义"))
    assert response.status_code == 200
    updated = response.json()["card"]
    assert (updated["knowledge_id"], updated["revision"], updated["status"]) == (knowledge_id, 2, "DRAFT")
    assert updated["payload"]["formula_sql"] == "SUM(amount)"
    assert updated["evidence_refs"] == original["nested"]["cards"][0]["evidence_refs"]
    saved = yaml.safe_load(source.read_text(encoding="utf-8"))
    assert saved["package_metadata"] == original["package_metadata"]
    assert saved["table_knowledge"] == original["table_knowledge"]
    assert saved["nested"]["cards"][1:] == original["nested"]["cards"][1:]
    assert saved["nested"]["cards"][0]["custom_metadata"] == original["nested"]["cards"][0]["custom_metadata"]
    assert current_knowledge.KNOWLEDGE_CARDS[knowledge_id].content["title"] == "修改后的标题"
    current_knowledge.reload_knowledge(root)
    assert client.get(f"/api/knowledge-cards/{knowledge_id}").json()["card"]["revision"] == 2


@pytest.mark.parametrize("kind", ["metric", "glossary_term"])
def test_create_draft_and_read_after_reload(client, knowledge_workspace, kind):
    from knowledge_runtime import current_knowledge

    root, _, _ = knowledge_workspace
    response = client.post("/api/knowledge-cards", json=create_body(kind))
    assert response.status_code == 201
    created = response.json()["card"]
    assert created["knowledge_id"].startswith(f"{kind}.manual.")
    assert (created["status"], created["revision"], created["evidence_refs"]) == ("DRAFT", 1, [])
    assert (root / created["source_file"]).is_file()
    current_knowledge.reload_knowledge(root)
    detail = client.get(f"/api/knowledge-cards/{created['knowledge_id']}")
    assert detail.json()["card"]["definition"] == "手工业务定义"
    assert created["knowledge_id"] in current_knowledge.KNOWLEDGE_CARDS


def test_physical_metadata_edit_preserves_payload_and_rejects_definition(client, knowledge_workspace):
    _, source, _ = knowledge_workspace
    before = load_knowledge_cards(source.parent)["column.example.amount"].content["payload"]
    response = client.patch("/api/knowledge-cards/column.example.amount", json=edit_body())
    assert response.status_code == 200
    assert response.json()["card"]["payload"] == before
    response = client.patch("/api/knowledge-cards/column.example.amount", json=edit_body(expected_revision=2, definition="修改物理字段"))
    assert response.status_code == 400


def test_relations_update_graph_and_show_incoming_links(client, knowledge_workspace):
    from knowledge_runtime import current_knowledge

    body = {"source_id": "metric.example.revenue", "target_id": "column.example.amount", "relation": "sourced_from", "expected_revision": 1}
    response = client.post("/api/knowledge-relations", json=body)
    assert response.status_code == 200
    assert response.json()["card"]["revision"] == 2
    assert response.json()["card"]["status"] == "DRAFT"
    incoming = client.get("/api/knowledge-cards/column.example.amount").json()["card"]["relations"]
    assert any(edge["source_id"] == body["source_id"] and edge["relation"] == "sourced_from" for edge in incoming)
    assert any(edge["relation"] == "sourced_from" for edge in current_knowledge.KNOWLEDGE_NAVIGATION_GRAPH["edges"])
    duplicate = client.post("/api/knowledge-relations", json={**body, "expected_revision": 2})
    assert duplicate.status_code == 409
    requires = client.post("/api/knowledge-relations", json={**body, "source_id": "query_recipe.example.report", "relation": "requires"})
    assert requires.status_code == 200


@pytest.mark.parametrize("changes,status", [
    ({"target_id": "metric.example.revenue"}, 400),
    ({"target_id": "metric.example.missing"}, 404),
    ({"target_id": "table.other.foreign"}, 400),
    ({"target_id": "../outside"}, 400),
    ({"relation": "invented"}, 422),
    ({"relation": "requires"}, 400),
    ({"source_id": "table.example.orders", "relation": "sourced_from"}, 400),
])
def test_invalid_relations_do_not_write(client, knowledge_workspace, changes, status):
    _, source, _ = knowledge_workspace
    original = source.read_bytes()
    body = {"source_id": "metric.example.revenue", "target_id": "column.example.amount", "relation": "related_to", "expected_revision": 1}
    assert client.post("/api/knowledge-relations", json={**body, **changes}).status_code == status
    assert source.read_bytes() == original


def test_stale_revision_active_run_and_invalid_fields_do_not_write(client, knowledge_workspace, monkeypatch):
    from api import configuration_app

    _, source, _ = knowledge_workspace
    original = source.read_bytes()
    url = "/api/knowledge-cards/metric.example.revenue"
    assert client.patch(url, json=edit_body(expected_revision=2)).status_code == 409
    assert client.patch(url, json=edit_body(formula_sql="DROP TABLE orders")).status_code == 422
    assert client.patch(url, json=edit_body(title="  ")).status_code == 422
    assert client.patch(url, json=edit_body(expected_revision=True)).status_code == 422
    assert client.post("/api/knowledge-cards", json=create_body(database_id="fabricated")).status_code == 400
    monkeypatch.setattr(configuration_app, "has_active_runs", lambda: True)
    assert client.patch(url, json=edit_body()).status_code == 409
    assert client.post("/api/knowledge-cards", json=create_body()).status_code == 409
    assert client.post("/api/knowledge-relations", json={"source_id": "metric.example.revenue", "target_id": "column.example.amount", "relation": "related_to", "expected_revision": 1}).status_code == 409
    assert source.read_bytes() == original


def test_permission_and_resource_guards(client, knowledge_workspace, monkeypatch):
    from api import configuration_app

    _, source, _ = knowledge_workspace
    original = source.read_bytes()
    for login_id in ("analyst-b", "viewer-c"):
        headers = {"X-Dev-User": login_id}
        assert client.patch("/api/knowledge-cards/metric.example.revenue", json=edit_body(), headers=headers).status_code == 403
        assert client.post("/api/knowledge-cards", json=create_body(), headers=headers).status_code == 403
        assert client.post("/api/knowledge-relations", json={}, headers=headers).status_code == 403
        assert client.get("/api/knowledge-cards/metric.example.revenue", headers=headers).status_code == 409
    resolve = configuration_app.current_user_from_request
    monkeypatch.setattr(configuration_app, "current_user_from_request", lambda request: replace(resolve(request), resources_ready=False))
    assert client.post("/api/knowledge-cards", json=create_body()).status_code == 409
    assert source.read_bytes() == original


def test_reload_failure_restores_exact_bytes_and_runtime(client, knowledge_workspace, monkeypatch):
    from api import configuration_app
    from knowledge_runtime import current_knowledge

    root, source, _ = knowledge_workspace
    original = source.read_bytes()
    original_cards = current_knowledge.KNOWLEDGE_CARDS
    def fail_reload(_):
        raise ValueError("reload failed")
    monkeypatch.setattr(configuration_app, "_refresh_knowledge_runtime", fail_reload)
    assert client.patch("/api/knowledge-cards/metric.example.revenue", json=edit_body(definition="新定义")).status_code == 500
    assert source.read_bytes() == original
    assert current_knowledge.KNOWLEDGE_CARDS is original_cards
    assert client.post("/api/knowledge-cards", json=create_body()).status_code == 500
    assert list(root.iterdir()) == [source]


def test_duplicate_ids_and_bad_shape_are_rejected(knowledge_workspace):
    root, source, original = knowledge_workspace
    broken = deepcopy(original)
    broken["nested"]["cards"].append(deepcopy(broken["table_knowledge"]))
    source.write_text(yaml.safe_dump(broken), encoding="utf-8")
    with pytest.raises(KnowledgeEditError):
        KnowledgeEditor(root, lambda _: None)
    broken = deepcopy(original)
    broken["table_knowledge"]["payload"] = ["not an object"]
    source.write_text(yaml.safe_dump(broken), encoding="utf-8")
    with pytest.raises(KnowledgeEditError):
        KnowledgeEditor(root, lambda _: None)


@pytest.mark.parametrize("reference", ["column.example.amount", 42])
def test_wrong_link_target_type_or_shape_is_rejected(knowledge_workspace, reference):
    root, source, original = knowledge_workspace
    original["nested"]["cards"][3]["payload"]["table_id"] = reference
    source.write_text(yaml.safe_dump(original), encoding="utf-8")
    with pytest.raises(KnowledgeEditError):
        KnowledgeEditor(root, lambda _: None)


def test_path_confinement_and_unknown_ids(client, knowledge_workspace, tmp_path):
    root, source, _ = knowledge_workspace
    editor = KnowledgeEditor(root, lambda _: None)
    original = source.read_bytes()
    with pytest.raises(KnowledgeEditError, match="路径"):
        editor._save("metric.example.revenue", deepcopy(editor.cards["metric.example.revenue"].content), tmp_path / "outside.yaml")
    assert not (tmp_path / "outside.yaml").exists()
    assert client.get("/api/knowledge-cards/metric.example.missing").status_code == 404
    assert client.get("/api/knowledge-cards/metric/unsafe").status_code == 400
    assert source.read_bytes() == original


def test_symlink_outside_root_is_rejected(knowledge_workspace, tmp_path):
    root, _, original = knowledge_workspace
    outside = tmp_path / "outside.yaml"
    outside.write_text(yaml.safe_dump(original), encoding="utf-8")
    try:
        (root / "outside-link.yaml").symlink_to(outside)
    except OSError:
        pytest.skip("Symbolic links unavailable for this Windows account")
    with pytest.raises(KnowledgeEditError, match="路径"):
        KnowledgeEditor(root, lambda _: None)
