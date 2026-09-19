<template>
  <div class="page">
    <h1>编辑对话</h1>
    <p class="lede">改 jsonl 里的句子，或删掉整段会话。进行中的通话请先挂断。</p>
    <p v-if="error" class="err">{{ error }}</p>
    <p v-if="notice" class="ok">{{ notice }}</p>
    <p v-if="jobs.running.length" class="lede">
      出题在后台进行，切走页面不会中断。顶部横幅会显示进度和结果。
    </p>
    <div class="card" style="margin-bottom:16px">
      <table class="table">
        <thead>
          <tr><th>会话</th><th>模式</th><th>时间</th><th></th></tr>
        </thead>
        <tbody>
          <tr v-for="item in sessions" :key="item.session_id">
            <td>{{ item.session_id.slice(0, 8) }}</td>
            <td>{{ item.mode }}</td>
            <td>{{ item.created_at }}</td>
            <td class="row">
              <button class="btn ghost" @click="open(item.session_id)">打开</button>
              <button class="btn ghost" :disabled="jobs.isBusy(item.session_id)" @click="generate(item.session_id)">
                {{ jobs.isBusy(item.session_id) ? "出题中…" : "出题" }}
              </button>
              <button class="btn danger" @click="remove(item.session_id)">删除</button>
              <span
                v-if="jobs.jobFor(item.session_id)"
                class="lede"
                style="max-width:220px"
              >{{ jobs.jobFor(item.session_id)?.message }}</span>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
    <div v-if="records.length" class="card">
      <h3>记录 {{ current }}</h3>
      <div v-for="row in records" :key="row.seq" style="margin:12px 0">
        <div class="lede">#{{ row.seq }} {{ row.type }} {{ row.role || "" }}</div>
        <textarea
          v-if="row.type === 'utterance'"
          v-model="row.text"
          @change="saveUtterance(row)"
          rows="2"
          style="width:100%"
        />
        <textarea
          v-else-if="row.type === 'scene'"
          :value="JSON.stringify(row.scene, null, 2)"
          @change="saveScene(row, ($event.target as HTMLTextAreaElement).value)"
          rows="6"
          style="width:100%"
        />
        <pre v-else style="font-size:12px;opacity:.7">{{ JSON.stringify(row) }}</pre>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api } from "../api";
import { useJobsStore } from "../stores/jobs";

type SessionRow = { session_id: string; mode: string; created_at: string };
type RecordRow = {
  seq: number;
  type: string;
  role?: string;
  text?: string;
  scene?: Record<string, string>;
};

const jobs = useJobsStore();
const sessions = ref<SessionRow[]>([]);
const records = ref<RecordRow[]>([]);
const current = ref("");
const error = ref("");
const notice = ref("");

async function load() {
  try {
    const data = await api<{ sessions: SessionRow[] }>("/v1/sessions");
    sessions.value = data.sessions;
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err);
  }
}

async function open(id: string) {
  error.value = "";
  current.value = id;
  try {
    const data = await api<{ records: RecordRow[] }>(`/v1/sessions/${id}/records`);
    records.value = data.records;
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err);
  }
}

async function saveUtterance(row: RecordRow) {
  await api(`/v1/sessions/${current.value}/records/${row.seq}`, {
    method: "PATCH",
    body: JSON.stringify({ text: row.text }),
  });
}

async function saveScene(row: RecordRow, raw: string) {
  const scene = JSON.parse(raw);
  await api(`/v1/sessions/${current.value}/records/${row.seq}`, {
    method: "PATCH",
    body: JSON.stringify({ scene }),
  });
  row.scene = scene;
}

async function remove(id: string) {
  if (!confirm("删除整段会话？")) return;
  await api(`/v1/sessions/${id}`, { method: "DELETE" });
  if (current.value === id) {
    current.value = "";
    records.value = [];
  }
  await load();
}

async function generate(id: string) {
  error.value = "";
  notice.value = "";
  try {
    await jobs.startGenerate(id);
    notice.value = jobs.jobFor(id)?.message || "已开始出题";
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err);
    notice.value = "";
  }
}

onMounted(load);
</script>
