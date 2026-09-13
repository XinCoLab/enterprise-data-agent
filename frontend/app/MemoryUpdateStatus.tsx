export type MemoryUpdate = {
  tool_call_id: string;
  status: "updating" | "pending" | "succeeded" | "failed" | "unknown" | "unchanged";
};

export default function MemoryUpdateStatus({ updates = [] }: { updates?: MemoryUpdate[] }) {
  if (!updates.length) return null;
  const failed = updates.some((update) => update.status === "failed");
  const updating = updates.some((update) => update.status === "updating");
  const unknown = updates.some((update) => update.status === "unknown");
  const pending = updates.some((update) => update.status === "pending");
  const succeeded = updates.some((update) => update.status === "succeeded");
  const label = failed ? (updates.length > 1 ? "记忆未全部更新成功" : "记忆更新失败")
    : updating ? "正在更新记忆"
      : unknown ? "记忆更新待确认"
        : pending ? "记忆已提交，待确认"
          : succeeded ? "记忆已更新" : "本次未新增记忆";

  return <div className={`memory-update-status${failed ? " failed" : ""}`} role="status" aria-live="polite" aria-atomic="true">
    <span className="memory-update-icon" aria-hidden="true" />
    <span>{label}</span>
  </div>;
}
