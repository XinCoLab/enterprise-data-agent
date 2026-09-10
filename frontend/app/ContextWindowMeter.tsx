import { useId } from "react";

export type ContextUsage = {
  input_tokens: number | null;
  context_window: number | null;
  model: string;
  source: "provider" | "unavailable";
};

export default function ContextWindowMeter({ usage, model, capacity }: {
  usage: ContextUsage | null;
  model: string;
  capacity?: number | null;
}) {
  const tooltipId = useId();
  const current = usage?.model === model ? usage : null;
  const limit = current?.context_window ?? capacity;
  const used = current?.source === "provider" ? current.input_tokens : null;
  const known = typeof used === "number" && Number.isFinite(used) && used >= 0
    && typeof limit === "number" && Number.isFinite(limit) && limit > 0;
  const percentage = known ? used / limit * 100 : null;
  const progress = percentage === null ? 0 : Math.min(100, percentage);
  const label = percentage === null ? "—" : percentage > 0 && percentage < 1 ? "<1%" : `${Math.round(percentage)}%`;
  const tone = percentage === null ? "unknown" : percentage >= 95 ? "critical" : percentage >= 80 ? "warning" : "normal";
  const format = (value: number) => value.toLocaleString("zh-CN");
  const description = known
    ? `上下文窗口已用 ${format(used)} / ${format(limit)} Token，${percentage!.toFixed(1)}%。最近一次模型输入，不含尚未发送的文字。`
    : "上下文用量尚未获取";

  return <div className={`context-meter ${tone}`}>
    <div className="context-meter-trigger" tabIndex={0} role="progressbar"
      aria-label="上下文窗口容量" aria-valuemin={0} aria-valuemax={100}
      aria-valuenow={known ? progress : undefined} aria-valuetext={description}
      aria-describedby={tooltipId}>
      <svg viewBox="0 0 36 36" aria-hidden="true">
        <circle className="context-meter-track" cx="18" cy="18" r="15" />
        <circle className="context-meter-fill" cx="18" cy="18" r="15" pathLength="100"
          strokeDasharray={`${progress} 100`} transform="rotate(-90 18 18)" />
      </svg>
      <span className="context-meter-value" aria-hidden="true">{label}</span>
    </div>
    <div className="context-meter-tooltip" role="tooltip" id={tooltipId}>
      {known ? `${format(used)} / ${format(limit)} tokens` : "暂无用量"}
    </div>
  </div>;
}
