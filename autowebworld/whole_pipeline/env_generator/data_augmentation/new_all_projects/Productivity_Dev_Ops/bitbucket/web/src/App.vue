<template>
  <div id="app" class="h-full bg-[#FAFBFC] text-[#172B4D] font-sans antialiased">
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
/* Global styles for Bitbucket theme */
body {
  margin: 0;
  padding: 0;
  background-color: #FAFBFC;
}

#app {
  min-height: 100vh;
}
</style>