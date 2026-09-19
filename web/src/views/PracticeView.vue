<template>
  <div ref="page" class="call-page">
    <div class="call-head">
      <h1>
        {{ call.status === "live" ? "通话中" : call.status === "connecting" ? "正在接通…" : "对练电话" }}
      </h1>
      <p style="opacity:.65;margin:6px 0 0;font-size:13px">
        {{ call.phase === "practice" ? "情景对练" : "先用语音说明场景与角色" }}
      </p>
      <div class="call-diag">
        <span
          v-for="row in call.diagRows"
          :key="row.id"
          class="diag-chip"
          :class="row.state"
          :title="row.text"
        >
          {{ row.label }} · {{ row.text }}
        </span>
      </div>
    </div>
    <div ref="stage" class="call-stage">
      <div v-if="!call.bubbles.length" style="opacity:.5;margin:auto;text-align:center">
        点底部绿色按钮拨打。接通后这里会显示状态和字幕。
      </div>
      <div
        v-for="item in call.bubbles"
        :key="item.id"
        class="bubble"
        :class="[item.role, { live: item.live }]"
      >
        {{ item.text }}
      </div>
    </div>
    <div class="call-dock">
      <textarea
        v-if="call.status !== 'live'"
        v-model="call.sceneDraft"
        placeholder="可选：先写一句场景，例如 ordering coffee at a cafe"
      />
      <button
        class="dial"
        :class="{ off: call.status === 'live' || call.status === 'connecting' }"
        @click="toggle"
      >
        {{ call.status === "live" ? "挂断" : call.status === "connecting" ? "取消" : "拨打" }}
      </button>
    </div>
    <Teleport to="body">
      <div
        class="avatar-pane"
        :class="{ dragging }"
        :style="paneStyle"
        @pointerdown="onDragStart"
      >
        <canvas ref="canvas" width="168" height="196"></canvas>
        <div v-if="!hasFrame" class="hint">{{ avatarHint }}</div>
      </div>
    </Teleport>
    <p v-if="call.error" class="err" style="position:absolute;left:24px;bottom:110px">{{ call.error }}</p>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, reactive, ref, watch } from "vue";
import { useCallStore } from "../stores/call";

const PANE_W = 168;
const PANE_H = 196;
const MARGIN = 8;

const call = useCallStore();
const page = ref<HTMLElement | null>(null);
const stage = ref<HTMLElement | null>(null);
const canvas = ref<HTMLCanvasElement | null>(null);
const hasFrame = ref(false);
const dragging = ref(false);
const userMoved = ref(false);
const avatarHint = ref("数字人");
const pos = reactive({ x: 24, y: 24 });
let dragOffset = { x: 0, y: 0 };
let frameSeq = 0;

const paneStyle = computed(() => ({
  left: `${pos.x}px`,
  top: `${pos.y}px`,
  right: "auto",
  bottom: "auto",
}));

function railInset() {
  return window.innerWidth > 800 ? 92 : 0;
}

function clampPos(x: number, y: number) {
  const minX = railInset() + MARGIN;
  const maxX = Math.max(minX, window.innerWidth - PANE_W - MARGIN);
  const maxY = Math.max(MARGIN, window.innerHeight - PANE_H - MARGIN);
  pos.x = Math.min(maxX, Math.max(minX, x));
  pos.y = Math.min(maxY, Math.max(MARGIN, y));
}

function placeDefault() {
  clampPos(window.innerWidth - PANE_W - 24, window.innerHeight - PANE_H - 24);
}

function scrollStageToBottom() {
  const el = stage.value;
  if (!el) return;
  el.scrollTop = el.scrollHeight;
}

watch(
  () => [call.bubbles.length, call.bubbles.at(-1)?.id, call.bubbles.at(-1)?.text],
  async () => {
    await nextTick();
    scrollStageToBottom();
  },
);

function onResize() {
  if (userMoved.value) {
    clampPos(pos.x, pos.y);
    return;
  }
  placeDefault();
}

function onDragStart(event: PointerEvent) {
  if (event.button !== 0) return;
  const pane = event.currentTarget as HTMLElement;
  dragging.value = true;
  userMoved.value = true;
  dragOffset = {
    x: event.clientX - pos.x,
    y: event.clientY - pos.y,
  };
  pane.setPointerCapture(event.pointerId);
  pane.addEventListener("pointermove", onDragMove);
  pane.addEventListener("pointerup", onDragEnd);
  pane.addEventListener("pointercancel", onDragEnd);
}

function onDragMove(event: PointerEvent) {
  if (!dragging.value) return;
  clampPos(event.clientX - dragOffset.x, event.clientY - dragOffset.y);
}

function onDragEnd(event: PointerEvent) {
  dragging.value = false;
  const pane = event.currentTarget as HTMLElement | null;
  pane?.removeEventListener("pointermove", onDragMove);
  pane?.removeEventListener("pointerup", onDragEnd);
  pane?.removeEventListener("pointercancel", onDragEnd);
}

async function drawFrame(data: ArrayBuffer | Blob) {
  const seq = ++frameSeq;
  const blob = data instanceof Blob ? data : new Blob([data], { type: "image/jpeg" });
  try {
    const bitmap = await createImageBitmap(blob);
    if (seq !== frameSeq) {
      bitmap.close();
      return;
    }
    const ctx = canvas.value?.getContext("2d");
    if (ctx && canvas.value) {
      ctx.drawImage(bitmap, 0, 0, canvas.value.width, canvas.value.height);
      hasFrame.value = true;
    }
    bitmap.close();
  } catch {
    avatarHint.value = "画面解码失败";
  }
}

async function toggle() {
  if (call.status === "connecting" || call.status === "live") {
    await call.hangup();
    hasFrame.value = false;
    avatarHint.value = "数字人";
    return;
  }
  if (call.status === "error") {
    await call.hangup();
    hasFrame.value = false;
    avatarHint.value = "数字人";
  }
  avatarHint.value = "加载中…";
  try {
    await call.start(call.sceneDraft.trim() || undefined);
  } catch {
    avatarHint.value = "数字人";
  }
}

onMounted(async () => {
  await nextTick();
  placeDefault();
  window.addEventListener("resize", onResize);
  call.setAvatarRenderer(drawFrame);
});
onUnmounted(() => {
  window.removeEventListener("resize", onResize);
  call.setAvatarRenderer(null);
  if (call.status === "live") void call.hangup();
  else if (call.status === "connecting") void call.hangup();
});
</script>
