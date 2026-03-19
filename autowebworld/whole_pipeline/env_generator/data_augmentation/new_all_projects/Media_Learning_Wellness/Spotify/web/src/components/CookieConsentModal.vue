<template>
  <div v-if="isVisible" class="fixed inset-0 z-[10000] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm transition-opacity duration-300">
    <div class="bg-[#282828] rounded-xl shadow-2xl max-w-md w-full p-6 text-white border border-[#3E3E3E] transform transition-all scale-100">
      <div class="flex items-center space-x-3 mb-4">
        <span class="text-3xl">🍪</span>
        <h2 class="text-xl font-bold">We Value Your Privacy</h2>
      </div>
      
      <p class="text-[#B3B3B3] mb-6 text-sm leading-relaxed">
        We use cookies to enhance your browsing experience, serve personalized ads or content, and analyze our traffic. By clicking "Accept All", you consent to our use of cookies.
      </p>
      
      <div class="flex space-x-3">
        <button 
          id="cookie-accept"
          class="flex-1 bg-[#1DB954] hover:bg-[#1ed760] text-black font-bold py-3 px-6 rounded-full transition-transform hover:scale-105 active:scale-95 text-sm uppercase tracking-widest"
          @click="acceptCookies"
        >
          Accept All
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'CookieConsentModal',
  setup() {
    const store = useSignatureStore()
    
    // Show if we are on HOME and consent is not given yet
    const isVisible = computed(() => {
      return store.currentPageId === 'HOME' && store.cookie_consent_given !== true
    })

    const acceptCookies = () => {
      store.cookie_consent_given = true
    }

    return {
      isVisible,
      acceptCookies
    }
  }
}
</script>