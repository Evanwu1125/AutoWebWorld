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
import { onMounted } from 'vue'
import { useSignatureStore } from './stores/signature'

export default {
  name: 'App',
  components: {
    CookieConsentModal,
    PermissionModal
  },
  setup() {
    const store = useSignatureStore()
    
    onMounted(() => {
      store.loadInitialData()
    })
  }
}
</script>

<style>
@import './style.css';

html, body {
  height: 100%;
  margin: 0;
  padding: 0;
}

#app {
  min-height: 100vh;
}
</style>