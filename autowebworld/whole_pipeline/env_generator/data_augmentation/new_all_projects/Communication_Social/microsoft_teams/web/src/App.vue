<template>
  <div id="app" class="h-full font-sans text-gray-900 bg-[#F5F5F5]">
    <router-view />
    <PermissionModal />
    <CookieConsentModal />
  </div>
</template>

<script>
import PermissionModal from './components/PermissionModal.vue'
import CookieConsentModal from './components/CookieConsentModal.vue'

export default {
  name: 'App',
  components: {
    PermissionModal,
    CookieConsentModal
  }
}
</script>

<style scoped>
#app {
  height: 100vh;
  width: 100vw;
}
</style>