import { defineStore } from "pinia";
import { api, wsUrl } from "../api";
import { PcmPlayer, startMicCapture } from "../pcm";

export type Bubble = {
  id: string;
  role: "user" | "assistant" | "system";
  text: string;
  live?: boolean;
};

type CreateSession = { session_id: string; mode: string; phase: string };
type AvatarRenderer = (data: ArrayBuffer | Blob) => void;
type PartState = "idle" | "pending" | "ok" | "fail";
type Part = { state: PartState; detail: string };

let avatarRenderer: AvatarRenderer | null = null;

const POST_TIMEOUT_MS = 10000;

const PART_LABEL: Record<string, string> = {
  voice: "语音API",
  rvc: "变声RVC",
  avatar: "数字人",
  mic: "麦克风",
};

function emptyParts(): Record<string, Part> {
  return {
    voice: { state: "idle", detail: "未开始" },
    rvc: { state: "idle", detail: "未开始" },
    avatar: { state: "idle", detail: "未开始" },
    mic: { state: "idle", detail: "未开始" },
  };
}

function isAbortError(err: unknown): boolean {
  return (
    (err instanceof DOMException && err.name === "AbortError") ||
    (err instanceof Error && err.name === "AbortError")
  );
}

export const useCallStore = defineStore("call", {
  state: () => ({
    sessionId: "" as string,
    phase: "",
    status: "idle" as "idle" | "connecting" | "live" | "error",
    sceneDraft: "",
    bubbles: [] as Bubble[],
    error: "",
    parts: emptyParts(),
    _startGen: 0,
    _abort: null as AbortController | null,
    _events: null as WebSocket | null,
    _audio: null as WebSocket | null,
    _avatar: null as WebSocket | null,
    _mic: null as { stop: () => void } | null,
    _player: null as PcmPlayer | null,
  }),
  getters: {
    diagRows(state) {
      return Object.keys(PART_LABEL).map((id) => ({
        id,
        label: PART_LABEL[id],
        state: state.parts[id]?.state || "idle",
        text: state.parts[id]?.detail || "",
      }));
    },
  },
  actions: {
    resetTranscript() {
      this.bubbles = [];
    },
    upsertLive(role: "user" | "assistant", text: string) {
      const last = this.bubbles[this.bubbles.length - 1];
      if (last && last.live && last.role === role) {
        last.text = text;
        return;
      }
      this.bubbles.push({ id: `${role}-live`, role, text, live: true });
    },
    commit(role: "user" | "assistant", text: string) {
      const last = this.bubbles[this.bubbles.length - 1];
      if (last && !last.live && last.role === role && last.text === text) return;
      this.bubbles = this.bubbles.filter((item) => !(item.live && item.role === role));
      this.bubbles.push({ id: `${role}-${this.bubbles.length}`, role, text });
    },
    pushSystem(text: string) {
      this.bubbles.push({ id: `sys-${this.bubbles.length}-${Date.now()}`, role: "system", text });
    },
    _setPart(name: string, state: PartState, detail: string) {
      this.parts = { ...this.parts, [name]: { state, detail } };
    },
    _resetParts() {
      this.parts = emptyParts();
    },
    _closeLocal() {
      this._mic?.stop();
      this._mic = null;
      this._player?.close();
      this._player = null;
      this._events?.close();
      this._audio?.close();
      this._avatar?.close();
      this._events = this._audio = this._avatar = null;
    },
    async start(scene?: string) {
      await this.hangup({ silent: true, bump: false });
      const gen = ++this._startGen;
      this.error = "";
      this.status = "connecting";
      this.resetTranscript();
      this._resetParts();
      this._setPart("voice", "pending", "正在创建会话…");
      this._setPart("rvc", "pending", "等待语音接通后加载");
      this._setPart("avatar", "pending", "后台加载本地模型");
      this._setPart("mic", "pending", "等待浏览器授权");
      this.pushSystem("正在创建会话…");
      const abort = new AbortController();
      this._abort = abort;
      const timer = window.setTimeout(() => abort.abort(), POST_TIMEOUT_MS);
      try {
        const created = await api<CreateSession>("/v1/sessions", {
          method: "POST",
          body: JSON.stringify({ mode: "web", scene: scene || null }),
          signal: abort.signal,
        });
        window.clearTimeout(timer);
        if (gen !== this._startGen) return;
        this.sessionId = created.session_id;
        this.phase = created.phase;
        this._ensurePageHide();
        this._bindEvents();
        this._player = new PcmPlayer();
        await this._player.ensure();
        if (gen !== this._startGen) return;
        this._bindAudio();
        this._bindAvatar();
        this.status = "live";
        this.pushSystem("会话已建立。下面四个状态会分别更新：语音 API / 变声 / 数字人 / 麦克风。");
      } catch (err) {
        if (gen !== this._startGen) return;
        if (isAbortError(err)) {
          this.status = "error";
          this.error = "创建会话超时（此时还没连语音 API / RVC / 数字人）";
          this._setPart("voice", "fail", this.error);
          this.pushSystem(this.error);
          const id = this.sessionId;
          this.sessionId = "";
          if (id) {
            const stopAbort = new AbortController();
            const stopTimer = window.setTimeout(() => stopAbort.abort(), 2000);
            try {
              await api(`/v1/sessions/${id}/stop`, { method: "POST", signal: stopAbort.signal });
            } catch {
              /* ignore */
            } finally {
              window.clearTimeout(stopTimer);
            }
          }
          return;
        }
        this.status = "error";
        this.error = err instanceof Error ? err.message : String(err);
        this._setPart("voice", "fail", this.error);
        this.pushSystem(`创建会话失败：${this.error}`);
        throw err;
      } finally {
        window.clearTimeout(timer);
        if (this._abort === abort) this._abort = null;
      }
    },
    _ensurePageHide() {
      if (typeof window === "undefined") return;
      if ((window as unknown as { __coachPageHide?: boolean }).__coachPageHide) return;
      (window as unknown as { __coachPageHide?: boolean }).__coachPageHide = true;
      window.addEventListener("pagehide", () => {
        const id = this.sessionId;
        if (!id) return;
        try {
          navigator.sendBeacon(`/v1/sessions/${id}/stop`);
        } catch {
          /* ignore */
        }
      });
    },
    _bindEvents() {
      const socket = new WebSocket(wsUrl(`/v1/sessions/${this.sessionId}/events?last_n=20`));
      this._events = socket;
      socket.onmessage = (event) => {
        if (this._events !== socket) return;
        const data = JSON.parse(event.data as string);
        const payload = data.payload || {};
        if (data.type === "asr.delta") this.upsertLive("user", payload.text || "");
        if (data.type === "asr.completed" && payload.text) this.commit("user", payload.text);
        if (data.type === "assistant.text_delta") this.upsertLive("assistant", payload.text || "");
        if (data.type === "assistant.text_done" && payload.text) this.commit("assistant", payload.text);
        if (data.type === "utterance" && payload.text && payload.role) {
          this.commit(payload.role, payload.text);
        }
        if (data.type === "scene.locked") this.phase = "practice";
        if (data.type === "session.ready") {
          this._setPart("voice", "ok", payload.detail || "语音通道就绪");
          this.pushSystem("语音通道就绪，可以说话");
        }
        if (data.type === "voice.connecting") {
          this._setPart("voice", "pending", payload.detail || "正在连接语音 API…");
          this.pushSystem(payload.detail || "正在连接语音 API…");
        }
        if (data.type === "voice.ready") {
          this._setPart("voice", "ok", payload.detail || "语音对话已接通");
          this.pushSystem(payload.detail || "语音对话已接通");
        }
        if (data.type === "voice.failed") {
          const text = payload.detail || payload.hint || payload.error || "语音 API 失败";
          this._setPart("voice", "fail", text);
          this.pushSystem(`语音API失败：${text}`);
        }
        if (data.type === "rvc.loading") {
          this._setPart("rvc", "pending", payload.detail || "正在加载变声…");
          this.pushSystem(payload.detail || "正在加载变声模型…");
        }
        if (data.type === "rvc.ready") {
          this._setPart("rvc", "ok", payload.detail || "变声已就绪");
          this.pushSystem("变声已就绪");
        }
        if (data.type === "rvc.skipped") {
          this._setPart("rvc", "ok", payload.detail || "未开启变声");
        }
        if (data.type === "rvc.failed") {
          const text = payload.detail || payload.hint || payload.message || "变声失败";
          this._setPart("rvc", "fail", text);
          this.pushSystem(`变声失败（将播放原声）：${text}`);
        }
        if (data.type === "rvc.converting") this.pushSystem("变声合成中，合成完才会出声和张口");
        if (data.type === "avatar.loading") {
          this._setPart("avatar", "pending", payload.detail || "数字人加载中…");
          this.pushSystem(payload.detail || "数字人加载中…");
        }
        if (data.type === "avatar.ready") {
          this._setPart("avatar", "ok", payload.detail || "数字人已就绪");
          this.pushSystem("数字人已就绪");
        }
        if (data.type === "avatar.failed") {
          const text = payload.detail || payload.hint || payload.message || "数字人失败";
          this._setPart("avatar", "fail", text);
          this.pushSystem(`数字人失败：${text}`);
        }
        if (data.type === "session.error") {
          const text = payload.detail || payload.hint || payload.message || "session error";
          this.pushSystem(text);
          if (payload.recoverable) {
            if (payload.component === "voice") this._setPart("voice", "pending", text);
            return;
          }
          this.error = text;
          this.status = "error";
          if (payload.component === "voice") this._setPart("voice", "fail", text);
        }
        if (data.type === "uplink.muted" && payload.reason === "max_speech") {
          this.pushSystem(payload.detail || "这一句太长，先按已识别内容让助手接话");
        }
        if (data.type === "session.ended") {
          if (this._events === socket) this.status = "idle";
        }
      };
      socket.onerror = () => {
        this.error = "事件通道连接失败";
        this.pushSystem(this.error);
      };
    },
    _bindAudio() {
      const socket = new WebSocket(wsUrl(`/v1/sessions/${this.sessionId}/audio`));
      socket.binaryType = "arraybuffer";
      this._audio = socket;
      socket.onmessage = (event) => {
        if (this._audio !== socket) return;
        if (typeof event.data === "string") return;
        void this._player?.push(event.data as ArrayBuffer);
      };
      socket.onopen = () => {
        void startMicCapture((pcm) => {
          if (socket.readyState === WebSocket.OPEN) socket.send(pcm);
        }).then((handle) => {
          this._mic = handle;
          this._setPart("mic", "ok", "麦克风已打开");
          this.pushSystem("麦克风已打开");
        }).catch((err) => {
          this.error = err instanceof Error ? err.message : String(err);
          this._setPart("mic", "fail", this.error);
          this.pushSystem(`麦克风失败：${this.error}（请在浏览器允许麦克风）`);
        });
      };
      socket.onerror = () => {
        this.error = "音频通道连接失败";
        this.pushSystem(this.error);
      };
    },
    setAvatarRenderer(fn: AvatarRenderer | null) {
      avatarRenderer = fn;
    },
    _bindAvatar() {
      const socket = new WebSocket(wsUrl(`/v1/sessions/${this.sessionId}/avatar`));
      socket.binaryType = "arraybuffer";
      this._avatar = socket;
      socket.onmessage = (event) => {
        if (this._avatar !== socket) return;
        if (typeof event.data === "string") {
          try {
            const msg = JSON.parse(event.data);
            if (msg.type === "disabled") this.pushSystem("数字人未启用");
          } catch {
            /* ignore */
          }
          return;
        }
        avatarRenderer?.(event.data as ArrayBuffer);
      };
      socket.onerror = () => this.pushSystem("数字人画面通道失败");
    },
    async hangup(opts: { silent?: boolean; bump?: boolean } = {}) {
      if (opts.bump !== false) this._startGen += 1;
      this._abort?.abort();
      this._abort = null;
      this._closeLocal();
      const id = this.sessionId;
      this.sessionId = "";
      this.status = "idle";
      if (!opts.silent) this.phase = "";
      if (id) {
        const stopAbort = new AbortController();
        const stopTimer = window.setTimeout(() => stopAbort.abort(), 2000);
        try {
          await api(`/v1/sessions/${id}/stop`, { method: "POST", signal: stopAbort.signal });
        } catch {
          /* ignore */
        } finally {
          window.clearTimeout(stopTimer);
        }
      }
    },
  },
});
