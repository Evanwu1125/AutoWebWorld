<template>
  <div v-if="isVisible" class="fixed inset-0 z-[10000] flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
    <div class="bg-white rounded-xl shadow-2xl max-w-md w-full p-6 animate-fade-in">
      <div class="text-4xl mb-4">🍪</div>
      <h2 class="text-2xl font-bold mb-2 text-gray-900">We Value Your Privacy</h2>
      <p class="text-gray-600 mb-6 leading-relaxed">
        We use cookies to enhance your browsing experience, serve personalized content, and analyze our traffic. By clicking 'Accept All', you consent to our use of cookies.
      </p>
      <div class="flex flex-col sm:flex-row gap-3">
        <button 
          id="cookie-accept"
          @click="acceptCookies"
          class="flex-1 bg-purple-600 hover:bg-purple-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors"
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
    
    // Only show on HOME and if not accepted yet
    const isVisible = computed(() => {
      return store.current_page_id === 'HOME' && !store.cookie_consent_given
    })

    const acceptCookies = () => {
      // Update the store to mark cookies as accepted
      store.cookie_consent_given = true
    }

    return {
      isVisible,
      acceptCookies
    }
  }
}
</script>

<style scoped>
.animate-fade-in {
  animation: fadeIn 0.3s ease-out;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>