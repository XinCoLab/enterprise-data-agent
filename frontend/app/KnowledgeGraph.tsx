"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type GraphNode = {
  ref: string;
  knowledge_id: string;
  knowledge_type: string;
  title: string;
};

type GraphEdge = {
  source: string;
  relation: string;
  target: string;
};

type GraphPayload = {
  nodes: GraphNode[];
  edges: GraphEdge[];
};

type SimulationNode = GraphNode & {
  x: number;
  y: number;
  vx: number;
  vy: number;
  degree: number;
};

type ViewTransform = { x: number; y: number; scale: number };

export type LiveKnowledgeTrace = {
  stage?: string;
  mode?: string;
  message?: string;
  activeIds: string[];
};

type KnowledgeGraphProps = {
  revision: number;
  live?: boolean;
  liveTrace?: LiveKnowledgeTrace | null;
};

let cachedGraphPayload: GraphPayload | null = null;

const NODE_COLORS: Record<string, string> = {
  database: "#2563eb",
  table: "#7c3aed",
  column: "#38bdf8",
  relationship: "#f59e0b",
  metric: "#10b981",
  glossary_term: "#ec4899",
};

const TYPE_LABELS: Record<string, string> = {
  database: "数据库",
  table: "表",
  column: "字段",
  relationship: "关系",
  metric: "指标",
  glossary_term: "术语",
};

function colorFor(type: string) {
  return NODE_COLORS[type] || "#64748b";
}

function stableNumber(value: string) {
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash * 31 + value.charCodeAt(index)) >>> 0;
  }
  return hash;
}

function createSimulation(payload: GraphPayload): SimulationNode[] {
  const degree = new Map(payload.nodes.map((node) => [node.ref, 0]));
  payload.edges.forEach((edge) => {
    degree.set(edge.source, (degree.get(edge.source) || 0) + 1);
    degree.set(edge.target, (degree.get(edge.target) || 0) + 1);
  });

  const types = [...new Set(payload.nodes.map((node) => node.knowledge_type))];
  const typeIndex = new Map(types.map((type, index) => [type, index]));

  return payload.nodes.map((node) => {
    const seed = stableNumber(node.ref);
    const groupIndex = typeIndex.get(node.knowledge_type) || 0;
    const groupAngle = (groupIndex / Math.max(types.length, 1)) * Math.PI * 2;
    const localAngle = ((seed % 360) / 180) * Math.PI;
    const localRadius = 24 + (seed % 58);
    return {
      ...node,
      x: Math.cos(groupAngle) * 105 + Math.cos(localAngle) * localRadius,
      y: Math.sin(groupAngle) * 76 + Math.sin(localAngle) * localRadius,
      vx: 0,
      vy: 0,
      degree: degree.get(node.ref) || 0,
    };
  });
}

export default function KnowledgeGraph({ revision, live = false, liveTrace = null }: KnowledgeGraphProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const shellRef = useRef<HTMLDivElement>(null);
  const drawRef = useRef<() => void>(() => undefined);
  const controllerRef = useRef<{ zoom: (factor: number) => void; reset: () => void; focus: (knowledgeIds: string[]) => void } | null>(null);
  const selectedRefValue = useRef<string | null>(null);
  const queryRef = useRef("");
  const liveTraceRef = useRef<LiveKnowledgeTrace | null>(liveTrace);
  const [payload, setPayload] = useState<GraphPayload | null>(cachedGraphPayload);
  const [selectedRef, setSelectedRef] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/api/knowledge-graph", { cache: "no-store" })
      .then(async (response) => {
        if (!response.ok) throw new Error("Knowledge 图加载失败");
        return response.json() as Promise<GraphPayload>;
      })
      .then((nextPayload) => {
        cachedGraphPayload = nextPayload;
        setError("");
        setPayload(nextPayload);
      })
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : "Knowledge 图加载失败"));
  }, [revision]);

  useEffect(() => {
    selectedRefValue.current = selectedRef;
    queryRef.current = query.trim().toLowerCase();
    drawRef.current();
  }, [query, selectedRef]);

  useEffect(() => {
    liveTraceRef.current = liveTrace;
    drawRef.current();
    const frame = window.requestAnimationFrame(() => {
      if (live) controllerRef.current?.focus(liveTrace?.activeIds || []);
    });
    return () => window.cancelAnimationFrame(frame);
  }, [live, liveTrace]);

  const selectedNode = useMemo(
    () => payload?.nodes.find((node) => node.ref === selectedRef) || null,
    [payload, selectedRef],
  );

  const selectedConnections = useMemo(() => {
    if (!payload || !selectedRef) return 0;
    return payload.edges.filter((edge) => edge.source === selectedRef || edge.target === selectedRef).length;
  }, [payload, selectedRef]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const shell = shellRef.current;
    if (!canvas || !shell || !payload) return;

    const canvasContext = canvas.getContext("2d");
    if (!canvasContext) return;
    const context: CanvasRenderingContext2D = canvasContext;

    const nodes = createSimulation(payload);
    const nodesByRef = new Map(nodes.map((node) => [node.ref, node]));
    const nodesByKnowledgeId = new Map(nodes.map((node) => [node.knowledge_id, node]));
    const edges = payload.edges
      .map((edge) => ({ ...edge, sourceNode: nodesByRef.get(edge.source), targetNode: nodesByRef.get(edge.target) }))
      .filter((edge) => edge.sourceNode && edge.targetNode) as (GraphEdge & { sourceNode: SimulationNode; targetNode: SimulationNode })[];
    const transform: ViewTransform = { x: 0, y: 0, scale: 1 };
    let width = 0;
    let height = 0;
    let hoveredNode: SimulationNode | null = null;
    let draggedNode: SimulationNode | null = null;
    let dragMoved = false;
    let isPanning = false;
    let lastPointer = { x: 0, y: 0 };
    let frame = 0;
    let animationFrame = 0;
    let viewTouched = false;

    const nodesAroundKnowledgeIds = (knowledgeIds: string[]) => {
      const activeRefs = new Set(
        knowledgeIds
          .map((knowledgeId) => nodesByKnowledgeId.get(knowledgeId)?.ref)
          .filter((nodeRef): nodeRef is string => Boolean(nodeRef)),
      );
      if (!activeRefs.size) return nodes;
      const visibleRefs = new Set(activeRefs);
      edges.forEach((edge) => {
        if (activeRefs.has(edge.source)) visibleRefs.add(edge.target);
        if (activeRefs.has(edge.target)) visibleRefs.add(edge.source);
      });
      return nodes.filter((node) => visibleRefs.has(node.ref));
    };

    const fitNodes = (targetNodes: SimulationNode[]) => {
      if (!width || !height || !targetNodes.length) return;
      const minX = Math.min(...targetNodes.map((node) => node.x));
      const maxX = Math.max(...targetNodes.map((node) => node.x));
      const minY = Math.min(...targetNodes.map((node) => node.y));
      const maxY = Math.max(...targetNodes.map((node) => node.y));
      const graphWidth = Math.max(maxX - minX, 80);
      const graphHeight = Math.max(maxY - minY, 80);
      const padding = live ? 100 : 70;
      transform.scale = Math.min(
        live ? 1.9 : 1.5,
        Math.max(
          .32,
          Math.min(
            (width - padding * 2) / graphWidth,
            (height - padding * 2) / graphHeight,
          ),
        ),
      );
      transform.x = width / 2 - ((minX + maxX) / 2) * transform.scale;
      transform.y = height / 2 - ((minY + maxY) / 2) * transform.scale;
      draw();
    };

    const focusKnowledgeIds = (knowledgeIds: string[]) => {
      fitNodes(nodesAroundKnowledgeIds(knowledgeIds));
    };

    const resetView = () => {
      viewTouched = false;
      focusKnowledgeIds(live ? (liveTraceRef.current?.activeIds || []) : []);
    };

    const resize = () => {
      const rect = shell.getBoundingClientRect();
      const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
      width = rect.width;
      height = rect.height;
      canvas.width = Math.round(width * pixelRatio);
      canvas.height = Math.round(height * pixelRatio);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
      resetView();
    };

    const screenPoint = (node: SimulationNode) => ({
      x: node.x * transform.scale + transform.x,
      y: node.y * transform.scale + transform.y,
    });

    const nodeAt = (x: number, y: number) => {
      for (let index = nodes.length - 1; index >= 0; index -= 1) {
        const node = nodes[index];
        const point = screenPoint(node);
        const radius = (6.5 + Math.min(7, Math.sqrt(node.degree) * 1.25)) * transform.scale + 6;
        if ((point.x - x) ** 2 + (point.y - y) ** 2 <= radius ** 2) return node;
      }
      return null;
    };

    function draw() {
      const now = performance.now();
      context.clearRect(0, 0, width, height);
      context.fillStyle = "#fbfcff";
      context.fillRect(0, 0, width, height);

      context.strokeStyle = "rgba(148,163,184,.12)";
      context.lineWidth = 1;
      const grid = 34 * transform.scale;
      if (grid > 14) {
        const offsetX = ((transform.x % grid) + grid) % grid;
        const offsetY = ((transform.y % grid) + grid) % grid;
        for (let x = offsetX; x < width; x += grid) {
          context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height); context.stroke();
        }
        for (let y = offsetY; y < height; y += grid) {
          context.beginPath(); context.moveTo(0, y); context.lineTo(width, y); context.stroke();
        }
      }

      const activeRef = selectedRefValue.current;
      const currentLiveTrace = live ? liveTraceRef.current : null;
      const activeLiveRefs = new Set(
        (currentLiveTrace?.activeIds || [])
          .map((knowledgeId) => nodesByKnowledgeId.get(knowledgeId)?.ref)
          .filter((nodeRef): nodeRef is string => Boolean(nodeRef)),
      );
      const liveScanning = Boolean(currentLiveTrace && activeLiveRefs.size === 0);
      const frontierRefs = new Set<string>();
      if (activeLiveRefs.size) {
        edges.forEach((edge) => {
          if (activeLiveRefs.has(edge.source)) frontierRefs.add(edge.target);
          if (activeLiveRefs.has(edge.target)) frontierRefs.add(edge.source);
        });
      }
      const livePulse = .5 + .5 * Math.sin(now / 330);
      edges.forEach((edge) => {
        const source = screenPoint(edge.sourceNode);
        const target = screenPoint(edge.targetNode);
        const active = activeRef && (edge.source === activeRef || edge.target === activeRef);
        const liveActive = currentLiveTrace && activeLiveRefs.size > 0 && (
          activeLiveRefs.has(edge.source)
          || activeLiveRefs.has(edge.target)
        );
        context.beginPath();
        context.moveTo(source.x, source.y);
        context.lineTo(target.x, target.y);
        context.setLineDash(liveActive ? [6, 7] : liveScanning ? [2, 11] : []);
        context.lineDashOffset = liveActive || liveScanning ? -now / 34 : 0;
        context.strokeStyle = liveActive
          ? `rgba(37,99,235,${.56 + livePulse * .28})`
          : liveScanning
            ? `rgba(37,99,235,${.12 + livePulse * .1})`
          : active
            ? "rgba(37,99,235,.62)"
            : currentLiveTrace && activeLiveRefs.size
              ? "rgba(148,163,184,.08)"
              : "rgba(100,116,139,.22)";
        context.lineWidth = liveActive ? 2 : active ? 1.8 : 0.8;
        context.stroke();
      });
      context.setLineDash([]);
      context.lineDashOffset = 0;

      const activeQuery = queryRef.current;
      nodes.forEach((node) => {
        const point = screenPoint(node);
        const matches = !activeQuery || `${node.title} ${node.knowledge_id}`.toLowerCase().includes(activeQuery);
        const selected = node.ref === activeRef;
        const hovered = node === hoveredNode;
        const liveActive = activeLiveRefs.has(node.ref);
        const liveFrontier = frontierRefs.has(node.ref);
        const scanPulse = .5 + .5 * Math.sin(now / 430 + (stableNumber(node.ref) % 12) * .42);
        const typeBoost = node.knowledge_type === "table" ? 1.8 : 0;
        const baseRadius = (6.5 + typeBoost + Math.min(7, Math.sqrt(node.degree) * 1.25)) * transform.scale;
        const radius = liveActive
          ? baseRadius * (1.42 + livePulse * .13)
          : liveFrontier
            ? baseRadius * 1.08
            : liveScanning
              ? baseRadius * (.92 + scanPulse * .16)
            : baseRadius;

        context.globalAlpha = currentLiveTrace && activeLiveRefs.size
          ? liveActive ? 1 : liveFrontier ? .68 : .1
          : liveScanning ? .28 + scanPulse * .72
          : matches ? 1 : 0.12;
        if (liveActive) {
          context.beginPath();
          context.arc(point.x, point.y, Math.max(8, radius + 7 + livePulse * 3), 0, Math.PI * 2);
          context.fillStyle = `rgba(37,99,235,${.08 + livePulse * .08})`;
          context.fill();
        }
        context.beginPath();
        context.arc(point.x, point.y, Math.max(2.6, radius), 0, Math.PI * 2);
        context.fillStyle = colorFor(node.knowledge_type);
        context.fill();
        if (selected || hovered) {
          context.strokeStyle = "#0f172a";
          context.lineWidth = 2;
          context.stroke();
        }

        const showLabel = selected || hovered || liveActive || node.knowledge_type === "table" || (activeQuery && matches);
        if (showLabel) {
          context.font = `${selected ? 600 : 500} ${selected ? 13 : 11}px Inter, system-ui, sans-serif`;
          const labelWidth = context.measureText(node.title).width;
          const labelX = point.x + radius + 7;
          const labelY = point.y + 4;
          context.fillStyle = "rgba(251,252,255,.9)";
          context.fillRect(labelX - 3, labelY - 12, labelWidth + 6, 16);
          context.fillStyle = "#273449";
          context.fillText(node.title, labelX, labelY);
        }
        context.globalAlpha = 1;
      });
    }

    const simulate = () => {
      if (frame < 190) {
        const cooling = 1 - frame / 210;
        for (let left = 0; left < nodes.length; left += 1) {
          const a = nodes[left];
          for (let right = left + 1; right < nodes.length; right += 1) {
            const b = nodes[right];
            const dx = a.x - b.x || 0.1;
            const dy = a.y - b.y || 0.1;
            const distanceSquared = Math.max(dx * dx + dy * dy, 64);
            const force = (48 * cooling) / distanceSquared;
            a.vx += dx * force;
            a.vy += dy * force;
            b.vx -= dx * force;
            b.vy -= dy * force;
          }
        }
        edges.forEach((edge) => {
          const dx = edge.targetNode.x - edge.sourceNode.x;
          const dy = edge.targetNode.y - edge.sourceNode.y;
          const distance = Math.max(Math.hypot(dx, dy), 1);
          const force = (distance - 42) * 0.016 * cooling;
          const fx = (dx / distance) * force;
          const fy = (dy / distance) * force;
          edge.sourceNode.vx += fx;
          edge.sourceNode.vy += fy;
          edge.targetNode.vx -= fx;
          edge.targetNode.vy -= fy;
        });
        nodes.forEach((node) => {
          node.vx += -node.x * 0.0018 * cooling;
          node.vy += -node.y * 0.0018 * cooling;
          node.vx *= 0.82;
          node.vy *= 0.82;
          if (node !== draggedNode) {
            node.x += node.vx;
            node.y += node.vy;
          }
        });
        frame += 1;
        if (live && !viewTouched && [1, 40, 100, 189].includes(frame)) {
          focusKnowledgeIds(liveTraceRef.current?.activeIds || []);
        }
      }
      draw();
      if (frame < 190 || draggedNode || isPanning || live) animationFrame = requestAnimationFrame(simulate);
    };

    const pointerPosition = (event: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      return { x: event.clientX - rect.left, y: event.clientY - rect.top };
    };

    const pointerDown = (event: PointerEvent) => {
      viewTouched = true;
      const point = pointerPosition(event);
      lastPointer = point;
      draggedNode = nodeAt(point.x, point.y);
      dragMoved = false;
      if (draggedNode) {
        selectedRefValue.current = draggedNode.ref;
        setSelectedRef(draggedNode.ref);
      } else {
        isPanning = true;
        selectedRefValue.current = null;
        setSelectedRef(null);
      }
      canvas.setPointerCapture(event.pointerId);
      draw();
    };

    const pointerMove = (event: PointerEvent) => {
      const point = pointerPosition(event);
      if (draggedNode) {
        draggedNode.x = (point.x - transform.x) / transform.scale;
        draggedNode.y = (point.y - transform.y) / transform.scale;
        draggedNode.vx = 0;
        draggedNode.vy = 0;
        dragMoved = true;
      } else if (isPanning) {
        transform.x += point.x - lastPointer.x;
        transform.y += point.y - lastPointer.y;
        dragMoved = true;
      } else {
        hoveredNode = nodeAt(point.x, point.y);
        canvas.style.cursor = hoveredNode ? "pointer" : "grab";
      }
      lastPointer = point;
      draw();
    };

    const pointerUp = (event: PointerEvent) => {
      if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
      draggedNode = null;
      isPanning = false;
      if (dragMoved) frame = Math.min(frame, 165);
      cancelAnimationFrame(animationFrame);
      animationFrame = requestAnimationFrame(simulate);
    };

    const wheel = (event: WheelEvent) => {
      event.preventDefault();
      viewTouched = true;
      const point = pointerPosition(event as unknown as PointerEvent);
      const previousScale = transform.scale;
      const nextScale = Math.min(2.8, Math.max(0.35, previousScale * Math.exp(-event.deltaY * 0.0012)));
      const worldX = (point.x - transform.x) / previousScale;
      const worldY = (point.y - transform.y) / previousScale;
      transform.scale = nextScale;
      transform.x = point.x - worldX * nextScale;
      transform.y = point.y - worldY * nextScale;
      draw();
    };

    controllerRef.current = {
      zoom(factor) {
        transform.scale = Math.min(2.8, Math.max(0.35, transform.scale * factor));
        draw();
      },
      reset: resetView,
      focus(knowledgeIds) {
        viewTouched = false;
        focusKnowledgeIds(knowledgeIds);
      },
    };
    drawRef.current = draw;

    const observer = new ResizeObserver(resize);
    observer.observe(shell);
    canvas.addEventListener("pointerdown", pointerDown);
    canvas.addEventListener("pointermove", pointerMove);
    canvas.addEventListener("pointerup", pointerUp);
    canvas.addEventListener("pointercancel", pointerUp);
    canvas.addEventListener("wheel", wheel, { passive: false });
    resize();
    animationFrame = requestAnimationFrame(simulate);

    return () => {
      observer.disconnect();
      cancelAnimationFrame(animationFrame);
      canvas.removeEventListener("pointerdown", pointerDown);
      canvas.removeEventListener("pointermove", pointerMove);
      canvas.removeEventListener("pointerup", pointerUp);
      canvas.removeEventListener("pointercancel", pointerUp);
      canvas.removeEventListener("wheel", wheel);
      drawRef.current = () => undefined;
      controllerRef.current = null;
    };
  }, [payload, live]);

  if (error) return <div className="knowledge-graph-error">{error}</div>;
  if (!payload) return <div className="knowledge-graph-loading">正在生成 Knowledge 图…</div>;

  return <section className={`knowledge-graph-panel ${live ? "live-knowledge-graph" : ""}`}>
    <div className="knowledge-graph-heading">
      <div><h2>{live ? "Knowledge Navigation" : "Knowledge Graph"}</h2><p>{live ? (liveTrace?.message || "正在定位业务知识") : `${payload.nodes.length} 个节点 · ${payload.edges.length} 条连接`}</p></div>
      {live ? <div className="live-knowledge-mode"><i />{liveTrace?.mode || "GLOBAL"}</div> : <div className="knowledge-graph-actions">
        <label className="knowledge-graph-search"><span>⌕</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="查找节点" /></label>
        <button type="button" onClick={() => controllerRef.current?.zoom(1.18)} aria-label="放大">＋</button>
        <button type="button" onClick={() => controllerRef.current?.zoom(0.84)} aria-label="缩小">−</button>
        <button type="button" onClick={() => controllerRef.current?.reset()}>复位</button>
      </div>}
    </div>
    <div className="knowledge-graph-shell" ref={shellRef}>
      <canvas ref={canvasRef} aria-label="Knowledge 节点关系图" />
      <div className="knowledge-graph-legend">
        {Object.entries(TYPE_LABELS).map(([type, label]) => <span key={type}><i style={{ background: colorFor(type) }} />{label}</span>)}
      </div>
      {selectedNode && <div className="knowledge-node-card">
        <div><i style={{ background: colorFor(selectedNode.knowledge_type) }} /><span>{TYPE_LABELS[selectedNode.knowledge_type] || selectedNode.knowledge_type}</span></div>
        <strong>{selectedNode.title}</strong>
        <code>{selectedNode.knowledge_id}</code>
        <small>{selectedConnections} 条连接</small>
      </div>}
    </div>
  </section>;
}
