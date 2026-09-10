"use client";

import { useEffect, useId, useMemo, useRef, useState, type ReactNode } from "react";
import { fetchJsonForUser } from "./api-client";
import "./knowledge-graph.css";

export type GraphNode = {
  ref: string;
  knowledge_id: string;
  knowledge_type: string;
  title: string;
  database_id?: string;
  summary?: string;
  status?: string;
  revision?: number;
  aliases?: string[];
};

export type GraphEdge = {
  source: string;
  relation: string;
  target: string;
};

export type GraphPayload = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  database_ids?: string[];
};

type SimulationNode = GraphNode & {
  x: number;
  y: number;
  vx: number;
  vy: number;
  degree: number;
};

type ViewTransform = { x: number; y: number; scale: number };

type SavedGraphView = {
  devUser: string;
  positions: Map<string, { x: number; y: number }>;
  transform: ViewTransform;
  width: number;
  height: number;
  viewTouched: boolean;
};

export type LiveKnowledgeTrace = {
  stage?: string;
  mode?: string;
  message?: string;
  activeIds: string[];
};

type KnowledgeGraphProps = {
  revision: number;
  devUser: string;
  live?: boolean;
  liveTrace?: LiveKnowledgeTrace | null;
  onSelectNode?: (node: GraphNode | null) => void;
  selectedKnowledgeId?: string | null;
  onGraphLoaded?: (payload: GraphPayload) => void;
  toolbarActions?: ReactNode;
};

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

function matchesNode(node: GraphNode, query: string) {
  return !query || `${node.title} ${node.knowledge_id} ${(node.aliases || []).join(" ")}`.toLowerCase().includes(query);
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

export default function KnowledgeGraph({ revision, devUser, live = false, liveTrace = null, onSelectNode, selectedKnowledgeId, onGraphLoaded, toolbarActions }: KnowledgeGraphProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const shellRef = useRef<HTMLDivElement>(null);
  const drawRef = useRef<() => void>(() => undefined);
  const controllerRef = useRef<{ zoom: (factor: number) => void; reset: () => void; focus: (knowledgeIds: string[]) => void; reveal: (knowledgeId: string) => void } | null>(null);
  const selectedIdRef = useRef<string | null>(null);
  const queryRef = useRef("");
  const selectionHandlerRef = useRef<(node: GraphNode | null) => void>(() => undefined);
  const onGraphLoadedRef = useRef(onGraphLoaded);
  const liveTraceRef = useRef<LiveKnowledgeTrace | null>(liveTrace);
  const savedViewRef = useRef<SavedGraphView | null>(null);
  const payloadUserRef = useRef(devUser);
  const focusSelectionRef = useRef<string | null>(null);
  const resultListRef = useRef<HTMLUListElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const searchId = useId();
  const [payload, setPayload] = useState<GraphPayload | null>(null);
  const [localSelectedId, setLocalSelectedId] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const [activeResult, setActiveResult] = useState(0);
  const [error, setError] = useState("");
  const effectiveSelectedId = live || selectedKnowledgeId === undefined ? localSelectedId : selectedKnowledgeId;

  useEffect(() => {
    onGraphLoadedRef.current = onGraphLoaded;
  }, [onGraphLoaded]);

  useEffect(() => {
    selectionHandlerRef.current = (node) => {
      if (live || selectedKnowledgeId === undefined) setLocalSelectedId(node?.knowledge_id || null);
      if (!live) onSelectNode?.(node);
    };
  }, [live, onSelectNode, selectedKnowledgeId]);

  useEffect(() => {
    let current = true;
    if (payloadUserRef.current !== devUser) {
      setPayload(null);
      setLocalSelectedId(null);
      payloadUserRef.current = devUser;
    }
    fetchJsonForUser<GraphPayload>("/api/knowledge-graph", devUser, { cache: "no-store" })
      .then((nextPayload) => {
        if (!current) return;
        setError("");
        setPayload(nextPayload);
        setActiveResult(0);
        onGraphLoadedRef.current?.(nextPayload);
      })
      .catch((reason: unknown) => {
        if (current) setError(reason instanceof Error ? reason.message : "知识图谱加载失败");
      });
    return () => { current = false; };
  }, [devUser, revision]);

  useEffect(() => {
    selectedIdRef.current = effectiveSelectedId;
    queryRef.current = query.trim().toLowerCase();
    if (!live && effectiveSelectedId) {
      if (focusSelectionRef.current === effectiveSelectedId) {
        controllerRef.current?.focus([effectiveSelectedId]);
        focusSelectionRef.current = null;
      } else {
        controllerRef.current?.reveal(effectiveSelectedId);
      }
    }
    drawRef.current();
  }, [effectiveSelectedId, live, payload, query]);

  useEffect(() => {
    liveTraceRef.current = liveTrace;
    drawRef.current();
    const frame = window.requestAnimationFrame(() => {
      if (live) controllerRef.current?.focus(liveTrace?.activeIds || []);
    });
    return () => window.cancelAnimationFrame(frame);
  }, [live, liveTrace]);

  const selectedNode = useMemo(
    () => payload?.nodes.find((node) => node.knowledge_id === effectiveSelectedId) || null,
    [payload, effectiveSelectedId],
  );

  const selectedConnections = useMemo(() => {
    if (!payload || !selectedNode) return 0;
    return payload.edges.filter((edge) => edge.source === selectedNode.ref || edge.target === selectedNode.ref).length;
  }, [payload, selectedNode]);

  const searchResults = useMemo(() => {
    const activeQuery = query.trim().toLowerCase();
    return (payload?.nodes || []).filter((node) => matchesNode(node, activeQuery))
      .sort((a, b) => a.title.localeCompare(b.title, "zh-CN"));
  }, [payload, query]);

  useEffect(() => {
    if (searchOpen) resultListRef.current?.children[activeResult]?.scrollIntoView({ block: "nearest" });
  }, [activeResult, searchOpen]);

  const chooseSearchResult = (node: GraphNode) => {
    focusSelectionRef.current = node.knowledge_id;
    selectionHandlerRef.current(node);
    if (node.knowledge_id === effectiveSelectedId) {
      controllerRef.current?.focus([node.knowledge_id]);
      focusSelectionRef.current = null;
    }
    setQuery("");
    setActiveResult(0);
    setSearchOpen(false);
    searchRef.current?.focus();
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    const shell = shellRef.current;
    if (!canvas || !shell || !payload) return;

    const canvasContext = canvas.getContext("2d");
    if (!canvasContext) return;
    const context: CanvasRenderingContext2D = canvasContext;

    const savedView = !live && savedViewRef.current?.devUser === devUser ? savedViewRef.current : null;
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const nodes = createSimulation(payload).map((node) => ({ ...node, ...savedView?.positions.get(node.knowledge_id) }));
    const nodesByRef = new Map(nodes.map((node) => [node.ref, node]));
    const nodesByKnowledgeId = new Map(nodes.map((node) => [node.knowledge_id, node]));
    const edges = payload.edges
      .map((edge) => ({ ...edge, sourceNode: nodesByRef.get(edge.source), targetNode: nodesByRef.get(edge.target) }))
      .filter((edge) => edge.sourceNode && edge.targetNode) as (GraphEdge & { sourceNode: SimulationNode; targetNode: SimulationNode })[];
    const transform: ViewTransform = savedView ? { ...savedView.transform } : { x: 0, y: 0, scale: 1 };
    let width = savedView?.width || 0;
    let height = savedView?.height || 0;
    let hoveredNode: SimulationNode | null = null;
    let draggedNode: SimulationNode | null = null;
    let pressedNode: SimulationNode | null = null;
    let pointerStart = { x: 0, y: 0 };
    let dragMoved = false;
    let isPanning = false;
    let lastPointer = { x: 0, y: 0 };
    let frame = savedView && nodes.every((node) => savedView.positions.has(node.knowledge_id)) ? 190 : 0;
    let animationFrame = 0;
    let viewTouched = savedView?.viewTouched || false;

    // New nodes join their real neighbours while existing knowledge keeps its position.
    if (savedView) {
      nodes.forEach((node) => {
        if (savedView.positions.has(node.knowledge_id)) return;
        const neighbours = edges.flatMap((edge) => edge.source === node.ref ? [edge.targetNode] : edge.target === node.ref ? [edge.sourceNode] : [])
          .filter((neighbour) => savedView.positions.has(neighbour.knowledge_id));
        if (!neighbours.length) return;
        const angle = (stableNumber(node.knowledge_id) % 360) * Math.PI / 180;
        node.x = neighbours.reduce((total, neighbour) => total + neighbour.x, 0) / neighbours.length + Math.cos(angle) * 45;
        node.y = neighbours.reduce((total, neighbour) => total + neighbour.y, 0) / neighbours.length + Math.sin(angle) * 45;
      });
    }

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
          live ? .32 : .08,
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
      const previousWidth = width;
      const previousHeight = height;
      width = rect.width;
      height = rect.height;
      canvas.width = Math.round(width * pixelRatio);
      canvas.height = Math.round(height * pixelRatio);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
      if (live || !viewTouched || !previousWidth || !previousHeight) resetView();
      else {
        transform.x += (width - previousWidth) / 2;
        transform.y += (height - previousHeight) / 2;
        draw();
      }
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
      const now = reducedMotion ? 0 : performance.now();
      context.clearRect(0, 0, width, height);
      context.fillStyle = "#fbfcff";
      context.fillRect(0, 0, width, height);

      context.strokeStyle = "rgba(148,163,184,.12)";
      context.lineWidth = 1;
      const grid = live ? 34 * transform.scale : Math.max(17, 34 * transform.scale);
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

      const activeRef = selectedIdRef.current ? nodesByKnowledgeId.get(selectedIdRef.current)?.ref : null;
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
        const matches = matchesNode(node, activeQuery);
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
          : selected || hovered || matches ? 1 : 0.12;
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

    const simulateStep = () => {
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
          if (node !== draggedNode && !savedView?.positions.has(node.knowledge_id)) {
            node.x += node.vx;
            node.y += node.vy;
          }
        });
        frame += 1;
      }
    };

    const simulate = () => {
      const previousFrame = frame;
      simulateStep();
      if (frame !== previousFrame && live && !viewTouched && [1, 40, 100, 189].includes(frame)) {
        focusKnowledgeIds(liveTraceRef.current?.activeIds || []);
      }
      draw();
      if (frame < 190 || draggedNode || isPanning || (live && !reducedMotion)) animationFrame = requestAnimationFrame(simulate);
    };

    const pointerPosition = (event: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      return { x: event.clientX - rect.left, y: event.clientY - rect.top };
    };

    const pointerDown = (event: PointerEvent) => {
      if (event.button !== 0) return;
      viewTouched = true;
      const point = pointerPosition(event);
      lastPointer = point;
      pointerStart = point;
      draggedNode = nodeAt(point.x, point.y);
      pressedNode = draggedNode;
      dragMoved = false;
      isPanning = !draggedNode;
      if (live) {
        selectedIdRef.current = draggedNode?.knowledge_id || null;
        selectionHandlerRef.current(draggedNode);
      }
      canvas.setPointerCapture(event.pointerId);
      canvas.style.cursor = draggedNode ? "grabbing" : "grab";
      draw();
    };

    const pointerMove = (event: PointerEvent) => {
      const point = pointerPosition(event);
      if (draggedNode) {
        draggedNode.x = (point.x - transform.x) / transform.scale;
        draggedNode.y = (point.y - transform.y) / transform.scale;
        draggedNode.vx = 0;
        draggedNode.vy = 0;
        dragMoved ||= Math.hypot(point.x - pointerStart.x, point.y - pointerStart.y) > 4;
      } else if (isPanning) {
        transform.x += point.x - lastPointer.x;
        transform.y += point.y - lastPointer.y;
        dragMoved ||= Math.hypot(point.x - pointerStart.x, point.y - pointerStart.y) > 4;
      } else {
        hoveredNode = nodeAt(point.x, point.y);
        canvas.style.cursor = hoveredNode ? "pointer" : "grab";
      }
      lastPointer = point;
      draw();
    };

    const pointerUp = (event: PointerEvent) => {
      if (!canvas.hasPointerCapture(event.pointerId)) return;
      if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
      if (!live && event.type !== "pointercancel" && !dragMoved) selectionHandlerRef.current(pressedNode);
      draggedNode = null;
      pressedNode = null;
      isPanning = false;
      canvas.style.cursor = hoveredNode ? "pointer" : "grab";
      if (live && dragMoved) {
        frame = Math.min(frame, 165);
        cancelAnimationFrame(animationFrame);
        animationFrame = requestAnimationFrame(simulate);
      }
      draw();
    };

    const pointerLeave = () => {
      if (draggedNode || isPanning) return;
      hoveredNode = null;
      draw();
    };

    const wheel = (event: WheelEvent) => {
      event.preventDefault();
      viewTouched = true;
      const point = pointerPosition(event as unknown as PointerEvent);
      const previousScale = transform.scale;
      const nextScale = Math.min(2.8, Math.max(live ? 0.35 : 0.08, previousScale * Math.exp(-event.deltaY * 0.0012)));
      const worldX = (point.x - transform.x) / previousScale;
      const worldY = (point.y - transform.y) / previousScale;
      transform.scale = nextScale;
      transform.x = point.x - worldX * nextScale;
      transform.y = point.y - worldY * nextScale;
      draw();
    };

    controllerRef.current = {
      zoom(factor) {
        viewTouched = true;
        transform.scale = Math.min(2.8, Math.max(live ? 0.35 : 0.08, transform.scale * factor));
        draw();
      },
      reset: resetView,
      focus(knowledgeIds) {
        viewTouched = !live;
        if (live) focusKnowledgeIds(knowledgeIds);
        else {
          const node = nodesByKnowledgeId.get(knowledgeIds[0]);
          if (!node) return;
          transform.x = width / 2 - node.x * transform.scale;
          transform.y = height / 2 - node.y * transform.scale;
          draw();
        }
      },
      reveal(knowledgeId) {
        const node = nodesByKnowledgeId.get(knowledgeId);
        if (!node) return;
        const point = screenPoint(node);
        if (point.x >= 65 && point.x <= width - 95 && point.y >= 45 && point.y <= height - 70) return;
        transform.x = width / 2 - node.x * transform.scale;
        transform.y = height / 2 - node.y * transform.scale;
        draw();
      },
    };
    drawRef.current = draw;

    const observer = new ResizeObserver(resize);
    observer.observe(shell);
    canvas.addEventListener("pointerdown", pointerDown);
    canvas.addEventListener("pointermove", pointerMove);
    canvas.addEventListener("pointerup", pointerUp);
    canvas.addEventListener("pointercancel", pointerUp);
    canvas.addEventListener("pointerleave", pointerLeave);
    canvas.addEventListener("wheel", wheel, { passive: false });
    if (reducedMotion || savedView) while (frame < 190) simulateStep();
    resize();
    if (live) focusKnowledgeIds(liveTraceRef.current?.activeIds || []);
    if (frame < 190 || (live && !reducedMotion)) animationFrame = requestAnimationFrame(simulate);

    return () => {
      savedViewRef.current = {
        devUser,
        positions: new Map(nodes.map((node) => [node.knowledge_id, { x: node.x, y: node.y }])),
        transform: { ...transform },
        width,
        height,
        viewTouched,
      };
      observer.disconnect();
      cancelAnimationFrame(animationFrame);
      canvas.removeEventListener("pointerdown", pointerDown);
      canvas.removeEventListener("pointermove", pointerMove);
      canvas.removeEventListener("pointerup", pointerUp);
      canvas.removeEventListener("pointercancel", pointerUp);
      canvas.removeEventListener("pointerleave", pointerLeave);
      canvas.removeEventListener("wheel", wheel);
      drawRef.current = () => undefined;
      controllerRef.current = null;
    };
  }, [payload, live, devUser]);

  if (error) return <div className="knowledge-graph-error">{error}</div>;
  if (!payload) return <div className="knowledge-graph-loading" role="status">正在加载知识图谱…</div>;

  return <section className={`knowledge-graph-panel ${live ? "live-knowledge-graph" : ""}`} aria-label={live ? "实时知识导航" : "知识图谱"}>
    <div className="knowledge-graph-heading">
      <div><h2>{live ? "Knowledge Navigation" : "Knowledge Graph"}</h2>{live && <p>{liveTrace?.message || "正在定位业务知识"}</p>}</div>
      {live ? <div className="live-knowledge-mode"><i />{liveTrace?.mode || "GLOBAL"}</div> : <div className="knowledge-graph-actions">
        <div className="kg-search-container" onBlur={(event) => {
          if (!event.currentTarget.contains(event.relatedTarget)) setSearchOpen(false);
        }}>
          <label className="knowledge-graph-search" htmlFor={`${searchId}-input`}>
            <span aria-hidden="true">⌕</span>
            <input ref={searchRef} id={`${searchId}-input`} value={query} onChange={(event) => { setQuery(event.target.value); setActiveResult(0); setSearchOpen(true); }}
              onFocus={() => setSearchOpen(true)} onKeyDown={(event) => {
                if (event.key === "ArrowDown" || event.key === "ArrowUp") {
                  event.preventDefault();
                  setSearchOpen(true);
                  setActiveResult((current) => searchResults.length ? (current + (event.key === "ArrowDown" ? 1 : -1) + searchResults.length) % searchResults.length : 0);
                } else if (event.key === "Enter" && searchOpen && searchResults[activeResult]) {
                  event.preventDefault();
                  chooseSearchResult(searchResults[activeResult]);
                } else if (event.key === "Escape") {
                  event.preventDefault();
                  setSearchOpen(false);
                }
              }}
              role="combobox" aria-label="搜索知识" aria-autocomplete="list" aria-expanded={searchOpen}
              aria-controls={`${searchId}-results`} aria-activedescendant={searchOpen && searchResults[activeResult] ? `${searchId}-result-${activeResult}` : undefined}
              placeholder="查找节点" autoComplete="off" />
            {query && <button type="button" className="kg-search-clear" aria-label="清空搜索" onClick={() => { setQuery(""); setActiveResult(0); searchRef.current?.focus(); }}>×</button>}
          </label>
          {searchOpen && <div className="kg-search-popover">
            <div className="kg-search-count" role="status">{searchResults.length} 条知识</div>
            <ul id={`${searchId}-results`} ref={resultListRef} className="kg-search-results" role="listbox" aria-label="知识搜索结果">
              {searchResults.map((node, index) => <li key={node.knowledge_id} id={`${searchId}-result-${index}`} role="option" tabIndex={-1} aria-selected={index === activeResult} className={index === activeResult ? "active" : ""}
                onMouseDown={(event) => event.preventDefault()} onMouseEnter={() => setActiveResult(index)} onClick={() => chooseSearchResult(node)} onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") { event.preventDefault(); chooseSearchResult(node); }
                }}>
                <i style={{ background: colorFor(node.knowledge_type) }} />
                <span><strong>{node.title}</strong><small>{node.knowledge_id}</small></span>
                <em>{TYPE_LABELS[node.knowledge_type] || node.knowledge_type}</em>
              </li>)}
            </ul>
            {!searchResults.length && <div className="kg-search-empty">没有匹配的知识</div>}
          </div>}
        </div>
        <button type="button" onClick={() => controllerRef.current?.zoom(1.18)} aria-label="放大">＋</button>
        <button type="button" onClick={() => controllerRef.current?.zoom(0.84)} aria-label="缩小">−</button>
        <button type="button" onClick={() => controllerRef.current?.reset()}>复位</button>
        {toolbarActions}
      </div>}
    </div>
    <div className="knowledge-graph-shell" ref={shellRef}>
      <canvas ref={canvasRef} aria-label={`知识关系图，${payload.nodes.length} 个节点，${payload.edges.length} 条连接${selectedNode ? `，已选择 ${selectedNode.title}` : ""}`} />
      {!payload.nodes.length && <div className="kg-canvas-empty">暂无知识</div>}
      <div className="knowledge-graph-legend">
        {Object.entries(TYPE_LABELS).map(([type, label]) => <span key={type}><i style={{ background: colorFor(type) }} />{label}</span>)}
      </div>
      {selectedNode && (live || !onSelectNode) && <div className="knowledge-node-card">
        <div><i style={{ background: colorFor(selectedNode.knowledge_type) }} /><span>{TYPE_LABELS[selectedNode.knowledge_type] || selectedNode.knowledge_type}</span></div>
        <strong>{selectedNode.title}</strong>
        <code>{selectedNode.knowledge_id}</code>
        <small>{selectedConnections} 条连接</small>
      </div>}
    </div>
  </section>;
}
