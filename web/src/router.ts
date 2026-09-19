import { createRouter, createWebHistory } from "vue-router";
import PracticeView from "./views/PracticeView.vue";
import ReviewView from "./views/ReviewView.vue";
import SessionsView from "./views/SessionsView.vue";
import LibraryView from "./views/LibraryView.vue";
import SettingsView from "./views/SettingsView.vue";
import SettingsAdvancedView from "./views/SettingsAdvancedView.vue";

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", name: "practice", component: PracticeView },
    { path: "/review", name: "review", component: ReviewView },
    { path: "/sessions", name: "sessions", component: SessionsView },
    { path: "/library", name: "library", component: LibraryView },
    { path: "/settings", name: "settings", component: SettingsView },
    { path: "/settings/advanced", name: "settings-advanced", component: SettingsAdvancedView },
  ],
});
