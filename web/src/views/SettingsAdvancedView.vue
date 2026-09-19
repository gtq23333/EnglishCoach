<template>
  <div class="page">
    <h1>高级设置</h1>
    <p class="lede">对照配置文件字段。api_key 只写不读，留空表示不改。</p>
    <p v-if="error" class="err">{{ error }}</p>
    <div class="card">
      <label class="field">配置 JSON
        <textarea v-model="text" rows="22" style="font-family:ui-monospace,monospace" />
      </label>
      <div class="row" style="margin-top:12px">
        <button class="btn accent" @click="save">写入</button>
        <router-link class="btn ghost" to="/settings">返回基础</router-link>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api } from "../api";

const text = ref("{}");
const error = ref("");

onMounted(async () => {
  try {
    const data = await api<{ settings: unknown }>("/v1/settings");
    text.value = JSON.stringify(data.settings, null, 2);
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err);
  }
});

async function save() {
  error.value = "";
  const patch = JSON.parse(text.value);
  const data = await api<{ settings: unknown; notes?: string[] }>("/v1/settings", {
    method: "PATCH",
    body: JSON.stringify(patch),
  });
  text.value = JSON.stringify(data.settings ?? patch, null, 2);
}
</script>
