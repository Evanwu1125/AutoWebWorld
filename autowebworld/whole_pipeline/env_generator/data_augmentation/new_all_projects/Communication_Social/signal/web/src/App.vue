<template>
  <div id="app" class="h-full bg-slate-50 font-sans text-slate-900">
    <router-view />
    <CookieConsentModal />
    <PermissionModal />
  </div>
</template>

<script>
import CookieConsentModal from './components/CookieConsentModal.vue'
import PermissionModal from './components/PermissionModal.vue'

export default {
  name: 'App',
  components: {
    CookieConsentModal,
    PermissionModal
  }
}
</script>

<style>
/* Global styles */
body {
  @apply bg-slate-50;
}
</style>