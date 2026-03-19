<template>
  <div id="app" class="h-full font-serif bg-white text-gray-900 antialiased selection:bg-green-100 selection:text-green-900">
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
/* Global Typography Overrides for "Medium" feel */
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,500;0,600;0,700;1,400;1,500&family=Inter:wght@300;400;500;600&display=swap');

:root {
  --font-serif: 'Lora', serif;
  --font-sans: 'Inter', sans-serif;
}

.font-serif {
  font-family: var(--font-serif);
}

.font-sans {
  font-family: var(--font-sans);
}

body {
  overflow-x: hidden;
}
</style>