<template>
  <div v-if="isOpen" class="fixed inset-0 z-[10000] flex items-end sm:items-center justify-center bg-black/50 backdrop-blur-sm p-4">
    <div class="bg-white rounded-xl shadow-2xl p-6 max-w-lg w-full transform transition-all">
      <div class="flex items-start gap-4">
        <div class="bg-blue-100 p-3 rounded-full shrink-0">
          <span class="text-2xl">🍪</span>
        </div>
        <div class="flex-1">
          <h3 class="text-lg font-bold text-gray-900 mb-2">We Value Your Privacy</h3>
          <p class="text-gray-600 text-sm leading-relaxed mb-6">
            We use cookies to enhance your browsing experience, serve personalized content, and analyze our traffic. By clicking "Accept All", you consent to our use of cookies.
          </p>
          <div class="flex gap-3 justify-end">
            <button class="px-4 py-2 text-gray-600 font-medium hover:bg-gray-100 rounded-lg transition-colors">
              Customize
            </button>
            <button 
              id="cookie-accept"
              @click="acceptCookies"
              class="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg shadow-lg shadow-blue-600/20 transition-all transform hover:-translate-y-0.5"
            >
              Accept All
            </button>
          </div>
        </div>
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
    
    // Show if cookie_consent_given is null
    const isOpen = computed(() => store.cookie_consent_given === null)
    
    const acceptCookies = () => {
      // FSM Effect is applied by the page action handler, but here we can just emit or handle directly if we want.
      // However, to strictly follow FSM, the page should handle the logic.
      // But since this is a component, we can let the parent handle it or directly update store if it's a global interceptor.
      // In this architecture, we update store directly as per instructions "Directly update Pinia store".
      store.cookie_consent_given = true
    }
    
    return {
      isOpen,
      acceptCookies
    }
  }
}
</script>