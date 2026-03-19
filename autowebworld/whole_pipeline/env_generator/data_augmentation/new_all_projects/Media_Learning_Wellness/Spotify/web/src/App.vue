<template>
  <div id="app" class="h-screen w-screen bg-[#121212] text-white overflow-hidden font-sans">
    <!-- Modals -->
    <CookieConsentModal />
    <PermissionModal />
    
    <!-- Main Router View -->
    <div class="h-full w-full">
      <router-view />
    </div>
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
/* Global Styles from style.txt */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap');

body {
  font-family: 'Inter', sans-serif;
  background-color: #121212;
  color: white;
}

/* Scrollbar Styling */
::-webkit-scrollbar {
  width: 12px;
}

::-webkit-scrollbar-track {
  background: #121212; 
}
 
::-webkit-scrollbar-thumb {
  background: #555; 
  border-radius: 6px;
}

::-webkit-scrollbar-thumb:hover {
  background: #888; 
}
</style>