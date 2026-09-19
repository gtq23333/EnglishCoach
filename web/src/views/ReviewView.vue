<template>
  <div class="page">
    <h1>针对性复习</h1>
    <p class="lede">选择题练搭配；对照卡练整句。选项顺序以组卷为准。</p>
    <div class="tabs">
      <button class="btn ghost" :class="{ active: tab === 'mcq' }" @click="tab = 'mcq'">选择题</button>
      <button class="btn ghost" :class="{ active: tab === 'pair' }" @click="tab = 'pair'">对照记忆</button>
    </div>
    <p v-if="error" class="err">{{ error }}</p>

    <section v-if="tab === 'mcq'" class="card">
      <div class="row" style="margin-bottom:12px">
        <label class="field">范围
          <select v-model="source">
            <option value="all">全部</option>
            <option value="wrong">错题</option>
          </select>
        </label>
        <label class="field">题数
          <input type="number" v-model.number="count" min="1" max="30" />
        </label>
        <button class="btn accent" :disabled="busy" @click="startQuiz">{{ busy ? "组卷中…" : "开始" }}</button>
      </div>
      <p v-if="quiz && quiz.total === 0" class="lede">题库是空的。先在「习题」里入库或从会话出题。</p>
      <div v-if="quiz">
        <p class="lede">{{ quiz.answered }} / {{ quiz.total }} 已答，对 {{ quiz.correct }}</p>
        <div v-for="item in quiz.items" :key="item.mcq_id" style="margin-bottom:18px">
          <pre style="white-space:pre-wrap;font-family:inherit">{{ item.stem }}</pre>
          <button
            v-for="opt in item.options"
            :key="opt"
            class="quiz-opt"
            :class="{
              picked: item.selected === opt,
              right: item.answered && opt === item.correct_answer,
              wrong: item.answered && item.selected === opt && !item.correct,
            }"
            :disabled="item.answered"
            @click="answer(item.mcq_id, opt)"
          >
            {{ opt }}
          </button>
          <p v-if="item.answered && item.analysis" class="lede">{{ item.analysis }}</p>
        </div>
      </div>
    </section>

    <section v-else class="card">
      <div class="row" style="margin-bottom:12px">
        <button class="btn ghost" @click="loadDue">刷新到期</button>
      </div>
      <div v-if="!card" class="lede">没有到期卡片。</div>
      <div v-else>
        <div class="pair-grid">
          <div class="card" style="box-shadow:none">
            <div class="lede">不佳表达</div>
            <p>{{ card.original_sentence }}</p>
          </div>
          <div class="card" style="box-shadow:none;cursor:pointer" @click="revealed = !revealed">
            <div class="lede">参考表达（单击切换遮挡）</div>
            <p :class="{ mask: !revealed }">{{ card.polished_sentence }}</p>
          </div>
        </div>
        <p v-if="revealed" class="lede">{{ card.analysis }}</p>
        <div class="row" style="margin-top:16px">
          <button class="btn danger" @click="grade('again')">again</button>
          <button class="btn" @click="grade('good')">good</button>
          <button class="btn accent" @click="grade('easy')">easy</button>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api } from "../api";

type QuizItem = {
  mcq_id: string;
  stem?: string;
  options: string[];
  answered: boolean;
  selected: string | null;
  correct: boolean | null;
  correct_answer?: string;
  analysis?: string;
};
type Quiz = { id: string; items: QuizItem[]; answered: number; correct: number; total: number };
type Card = {
  id: string;
  original_sentence: string;
  polished_sentence: string;
  analysis: string;
};

const tab = ref<"mcq" | "pair">("mcq");
const source = ref("all");
const count = ref(8);
const quiz = ref<Quiz | null>(null);
const card = ref<Card | null>(null);
const revealed = ref(false);
const error = ref("");
const busy = ref(false);

async function startQuiz() {
  error.value = "";
  busy.value = true;
  try {
    quiz.value = await api<Quiz>("/v1/quizzes", {
      method: "POST",
      body: JSON.stringify({ source: source.value, count: count.value }),
    });
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err);
  } finally {
    busy.value = false;
  }
}

async function answer(mcqId: string, selected: string) {
  if (!quiz.value) return;
  try {
    quiz.value = await api<Quiz>(`/v1/quizzes/${quiz.value.id}/answers`, {
      method: "POST",
      body: JSON.stringify({ mcq_id: mcqId, selected, elapsed_ms: 0 }),
    });
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err);
  }
}

async function loadDue() {
  error.value = "";
  try {
    const data = await api<{ cards: Card[] }>("/v1/reviews/due");
    card.value = data.cards[0] || null;
    revealed.value = false;
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err);
  }
}

async function grade(value: string) {
  if (!card.value) return;
  try {
    await api(`/v1/reviews/${card.value.id}/grade`, {
      method: "POST",
      body: JSON.stringify({ grade: value }),
    });
    await loadDue();
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err);
  }
}

onMounted(() => {
  void loadDue();
});
</script>
