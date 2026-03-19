<template>
  <div id="app" class="h-full font-sans antialiased text-gray-900 bg-gray-50">
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
/* Global Styles */
body {
  margin: 0;
  padding: 0;
  background-color: #F8F9FA;
}
</style>