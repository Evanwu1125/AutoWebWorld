<template>
  <div v-if="isVisible" class="fixed inset-0 z-[10000] flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
    <div class="bg-white rounded-xl shadow-2xl max-w-md w-full p-6 animate-fade-in">
      <div class="flex items-center gap-3 mb-4">
        <div class="text-4xl">🍪</div>
        <h2 class="text-xl font-bold text-gray-900">We Value Your Privacy</h2>
      </div>
      <p class="text-gray-600 mb-6 leading-relaxed">
        We use cookies to enhance your project management experience, serve personalized content, and analyze our traffic. By clicking 'Accept All', you consent to our use of cookies.
      </p>
      <div class="flex gap-3 justify-end">
        <button 
          id="cookie-decline"
          @click="decline"
          class="px-4 py-2 rounded-lg text-gray-600 hover:bg-gray-100 font-medium transition-colors"
        >
          Decline
        </button>
        <button 
          id="cookie-accept"
          @click="accept"
          class="px-6 py-2 rounded-lg bg-indigo-600 text-white hover:bg-indigo-700 font-medium shadow-lg hover:shadow-xl transition-all"
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
    
    // Check FSM logic: Only show if cookie_consent_given is null (not exists)
    // AND we are on HOME page (usually handled by parent, but modal can be smart)
    const isVisible = computed(() => {
      return store.currentPageId === 'HOME' && store.cookie_consent_given === null
    })

    const accept = () => {
      // FSM Effect: set cookie_consent_given = true
      store.cookie_consent_given = true
    }

    const decline = () => {
        // Optional path not strictly in FSM happy path but good UX
        store.cookie_consent_given = false
    }

    return {
      isVisible,
      accept,
      decline
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