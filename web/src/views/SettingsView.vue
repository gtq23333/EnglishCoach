<template>
  <div class="page">
    <h1>设置</h1>
    <p class="lede">基础项会写回 config.toml。密钥不会从接口读出。</p>
    <p v-if="error" class="err">{{ error }}</p>
    <p v-if="notes.length" class="ok">{{ notes.join("；") }}</p>
    <div class="card" style="display:grid;gap:14px;max-width:480px">
      <label class="field">音色
        <input v-model="form.voice" />
      </label>
      <label class="field">语速
        <input type="number" v-model.number="form.speed" />
      </label>
      <label class="field">音量
        <input type="number" v-model.number="form.loudness" />
      </label>
      <label class="row"><input type="checkbox" v-model="form.avatar" /> 数字人</label>
      <label class="row"><input type="checkbox" v-model="form.rvc" /> RVC 变声</label>
      <div class="row">
        <button class="btn accent" @click="save">保存</button>
        <router-link class="btn ghost" to="/settings/advanced">高级设置</router-link>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, reactive, ref } from "vue";
import { api } from "../api";

const form = reactive({ voice: "", speed: 0, loudness: 0, avatar: true, rvc: false });
const notes = ref<string[]>([]);
const error = ref("");

async function load() {
  const data = await api<{ settings: Record<string, any> }>("/v1/settings");
  const s = data.settings || {};
  form.voice = s.session?.voice || s.session?.speaker || "";
  form.speed = s.session?.speed ?? 0;
  form.loudness = s.session?.loudness ?? 0;
  form.avatar = s.avatar?.enabled !== false;
  form.rvc = Boolean(s.voice?.rvc?.enabled);
}

async function save() {
  error.value = "";
  notes.value = [];
  try {
    const data = await api<{ notes: string[] }>("/v1/settings", {
      method: "PATCH",
      body: JSON.stringify({
        session: { voice: form.voice, speed: form.speed, loudness: form.loudness },
        avatar: { enabled: form.avatar },
        voice: { rvc: { enabled: form.rvc } },
      }),
    });
    notes.value = data.notes?.length ? data.notes : ["已保存"];
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err);
  }
}

onMounted(() => {
  void load().catch((err) => {
    error.value = err instanceof Error ? err.message : String(err);
  });
});
</script>
