<template>
  <div id="app" class="h-full font-sans text-gray-900 antialiased">
    <router-view />
    
    <!-- Global Modals -->
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
/* Global transitions if needed */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>