import { defineStore } from "pinia";
import { api } from "../api";

export type ItemJob = {
  job_id: string;
  session_id: string;
  status: "running" | "done" | "error";
  stage?: string;
  slice?: number;
  total?: number;
  attempt?: number;
  last_error?: string;
  cases?: number;
  ingested?: { mcqs?: number; pairs?: number; cases?: number };
  ingest_error?: string;
  error?: string;
  message: string;
};

const STORAGE_KEY = "coach.itemgen.jobs";

function loadSaved(): ItemJob[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as ItemJob[];
    return Array.isArray(parsed) ? parsed.slice(-8) : [];
  } catch {
    return [];
  }
}

function persist(items: ItemJob[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(items.slice(-8)));
}

function describe(job: ItemJob): string {
  if (job.status === "running") {
    const slice = job.slice && job.total ? `（${job.slice}/${job.total}）` : "";
    if (job.stage === "llm") return `正在调用出题模型${slice}…`;
    if (job.stage === "parse_retry") {
      return `模型输出无法解析，正在重试${slice}…`;
    }
    if (job.stage === "slice" || job.stage === "filtered") {
      return `正在出题${slice}…`;
    }
    return "正在出题（要调 LLM，请稍候）…";
  }
  if (job.status === "error") {
    return job.error || job.last_error || "出题失败";
  }
  if (job.ingest_error) return `出题完成但入库失败：${job.ingest_error}`;
  const mcqs = job.ingested?.mcqs ?? 0;
  const pairs = job.ingested?.pairs ?? 0;
  const cases = job.cases ?? 0;
  if (cases === 0) {
    return "出题完成，但没有可用题目（会话太短、被质量门跳过，或模型拒绝出题）。";
  }
  return `出题完成：${cases} 个情境，入库选择题 ${mcqs}、对照句 ${pairs}。请到「习题」查看。`;
}

export const useJobsStore = defineStore("jobs", {
  state: () => ({
    items: loadSaved() as ItemJob[],
    bannerDismissed: "" as string,
    _timers: {} as Record<string, number>,
  }),
  getters: {
    running(): ItemJob[] {
      return this.items.filter((item) => item.status === "running");
    },
    banner(): { kind: "ok" | "err" | "run"; text: string; to?: string } | null {
      const active = this.running[0];
      if (active) {
        return { kind: "run", text: active.message, to: "/sessions" };
      }
      const latest = this.items[this.items.length - 1];
      if (!latest || latest.job_id === this.bannerDismissed) return null;
      if (latest.status === "error") {
        return { kind: "err", text: latest.message, to: "/sessions" };
      }
      if (latest.status === "done") {
        return { kind: "ok", text: latest.message, to: "/library" };
      }
      return null;
    },
  },
  actions: {
    isBusy(sessionId: string) {
      return this.items.some(
        (item) => item.session_id === sessionId && item.status === "running"
      );
    },
    jobFor(sessionId: string) {
      return [...this.items].reverse().find((item) => item.session_id === sessionId);
    },
    dismissBanner() {
      const latest = this.items[this.items.length - 1];
      this.bannerDismissed = latest?.job_id || "";
    },
    hydrate() {
      for (const job of this.items) {
        if (job.status === "running") this._poll(job.job_id);
      }
    },
    _upsert(job: ItemJob) {
      const index = this.items.findIndex((item) => item.job_id === job.job_id);
      if (index >= 0) this.items[index] = job;
      else this.items.push(job);
      persist(this.items);
    },
    async startGenerate(sessionId: string) {
      if (this.isBusy(sessionId)) return this.jobFor(sessionId);
      const created = await api<{ job_id: string; session_id: string; status: string }>(
        `/v1/sessions/${sessionId}/items/generate`,
        { method: "POST", body: "{}" }
      );
      const job: ItemJob = {
        job_id: created.job_id,
        session_id: sessionId,
        status: "running",
        message: "正在出题（要调 LLM，请稍候）…",
      };
      this.bannerDismissed = "";
      this._upsert(job);
      this._poll(created.job_id);
      return job;
    },
    _poll(jobId: string) {
      if (this._timers[jobId]) return;
      const tick = async () => {
        try {
          const status = await api<ItemJob>(`/v1/jobs/${jobId}`);
          const current = this.items.find((item) => item.job_id === jobId);
          const merged: ItemJob = {
            ...current,
            ...status,
            job_id: jobId,
            session_id: status.session_id || current?.session_id || "",
            status: (status.status as ItemJob["status"]) || "running",
            message: "",
          };
          merged.message = describe(merged);
          this._upsert(merged);
          if (merged.status === "running") {
            this._timers[jobId] = window.setTimeout(tick, 2000);
            return;
          }
        } catch (err) {
          const current = this.items.find((item) => item.job_id === jobId);
          if (current && current.status === "running") {
            this._upsert({
              ...current,
              status: "error",
              error: err instanceof Error ? err.message : String(err),
              message: err instanceof Error ? err.message : String(err),
            });
          }
        }
        delete this._timers[jobId];
      };
      this._timers[jobId] = window.setTimeout(tick, 400);
    },
  },
});
