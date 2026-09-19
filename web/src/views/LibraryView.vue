<template>
  <div class="page">
    <h1>编辑习题</h1>
    <p class="lede">改题干、答案、润色句；软删不会抹掉做题记录。出题入口在左侧「会话」。出完后点刷新。</p>
    <p v-if="error" class="err">{{ error }}</p>
    <div class="row" style="margin-bottom:12px">
      <button class="btn accent" @click="createCase">新增 case</button>
      <button class="btn ghost" @click="load">刷新</button>
    </div>
    <p v-if="!cases.length" class="lede">还没有题目。可从「会话」里点出题，或这里手动新增。</p>
    <div v-for="item in cases" :key="item.id" class="card" style="margin-bottom:12px">
      <div class="row">
        <strong>{{ item.id }}</strong>
        <span class="lede">{{ item.scenario }}</span>
        <button class="btn ghost" @click="open(item.id)">展开</button>
        <button class="btn danger" @click="removeCase(item.id)">删除</button>
      </div>
      <div v-if="opened?.id === item.id" style="margin-top:12px">
        <label class="field">场景
          <input v-model="opened.scenario" @change="patchCase" />
        </label>
        <h4>选择题</h4>
        <div v-for="q in opened.mcqs" :key="q.id" class="card" style="box-shadow:none;margin:8px 0">
          <label class="field">题干<textarea v-model="q.stem" rows="3" @change="patchMcq(q)" /></label>
          <label class="field">正确答案<input v-model="q.correct_answer" @change="patchMcq(q)" /></label>
          <button class="btn danger" @click="removeMcq(q.id)">删题</button>
        </div>
        <button class="btn ghost" @click="addMcq">加一道选择题</button>
        <h4>对照句</h4>
        <div v-for="p in opened.sentence_pairs" :key="p.id" class="card" style="box-shadow:none;margin:8px 0">
          <label class="field">原句<textarea v-model="p.original_sentence" @change="patchPair(p)" /></label>
          <label class="field">润色<textarea v-model="p.polished_sentence" @change="patchPair(p)" /></label>
          <button class="btn danger" @click="removePair(p.id)">删句</button>
        </div>
        <button class="btn ghost" @click="addPair">加一组对照</button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api } from "../api";

type Mcq = { id: string; stem: string; correct_answer: string };
type Pair = { id: string; original_sentence: string; polished_sentence: string };
type Case = { id: string; scenario: string; mcqs?: Mcq[]; sentence_pairs?: Pair[] };

const cases = ref<Case[]>([]);
const opened = ref<Case | null>(null);
const error = ref("");

async function load() {
  const data = await api<{ cases: Case[] }>("/v1/library/cases");
  cases.value = data.cases;
}

async function open(id: string) {
  opened.value = await api<Case>(`/v1/library/cases/${id}`);
}

async function createCase() {
  error.value = "";
  try {
    const row = await api<Case>("/v1/library/cases", {
      method: "POST",
      body: JSON.stringify({ scenario: "manual" }),
    });
    await load();
    await open(row.id);
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err);
  }
}

async function patchCase() {
  if (!opened.value) return;
  await api(`/v1/library/cases/${opened.value.id}`, {
    method: "PATCH",
    body: JSON.stringify({ scenario: opened.value.scenario }),
  });
}

async function patchMcq(q: Mcq) {
  await api(`/v1/library/mcqs/${q.id}`, {
    method: "PATCH",
    body: JSON.stringify({ stem: q.stem, correct_answer: q.correct_answer }),
  });
}

async function patchPair(p: Pair) {
  await api(`/v1/library/pairs/${p.id}`, {
    method: "PATCH",
    body: JSON.stringify({
      original_sentence: p.original_sentence,
      polished_sentence: p.polished_sentence,
    }),
  });
}

async function addMcq() {
  if (!opened.value) return;
  await api(`/v1/library/cases/${opened.value.id}/mcqs`, {
    method: "POST",
    body: JSON.stringify({ stem: "User: I <blank>.", correct_answer: "went", original_distractor: "go", additional_distractors: ["goed", "gone"] }),
  });
  await open(opened.value.id);
}

async function addPair() {
  if (!opened.value) return;
  await api(`/v1/library/cases/${opened.value.id}/pairs`, {
    method: "POST",
    body: JSON.stringify({ original_sentence: "", polished_sentence: "" }),
  });
  await open(opened.value.id);
}

async function removeMcq(id: string) {
  await api(`/v1/library/mcqs/${id}`, { method: "DELETE" });
  if (opened.value) await open(opened.value.id);
}

async function removePair(id: string) {
  await api(`/v1/library/pairs/${id}`, { method: "DELETE" });
  if (opened.value) await open(opened.value.id);
}

async function removeCase(id: string) {
  await api(`/v1/library/cases/${id}`, { method: "DELETE" });
  if (opened.value?.id === id) opened.value = null;
  await load();
}

onMounted(() => {
  void load().catch((err) => {
    error.value = err instanceof Error ? err.message : String(err);
  });
});
</script>
