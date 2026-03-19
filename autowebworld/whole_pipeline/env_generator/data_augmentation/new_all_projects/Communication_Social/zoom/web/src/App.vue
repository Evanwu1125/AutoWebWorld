<template>
  <div id="app" class="h-full font-sans text-slate-900 bg-white dark:bg-slate-900 dark:text-white">
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

<style scoped>
#app {
  min-height: 100vh;
  width: 100vw;
}
</style>