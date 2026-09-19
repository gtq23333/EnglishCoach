<template>
  <div class="shell">
    <nav class="rail">
      <div class="brand">Coach</div>
      <router-link class="core" to="/">对练</router-link>
      <router-link class="core" to="/review">复习</router-link>
      <div style="flex:1"></div>
      <router-link class="edge" to="/sessions">会话</router-link>
      <router-link class="edge" to="/library">习题</router-link>
      <router-link class="edge" to="/settings">设置</router-link>
    </nav>
    <div class="main">
      <div v-if="jobs.banner" class="job-banner" :class="jobs.banner.kind">
        <span>{{ jobs.banner.text }}</span>
        <router-link v-if="jobs.banner.to" :to="jobs.banner.to">查看</router-link>
        <button type="button" class="btn ghost" style="padding:4px 10px" @click="jobs.dismissBanner()">
          关闭
        </button>
      </div>
      <router-view />
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted } from "vue";
import { useJobsStore } from "../stores/jobs";

const jobs = useJobsStore();
onMounted(() => jobs.hydrate());
</script>
