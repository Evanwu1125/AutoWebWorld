<template>
  <div v-if="isVisible" class="fixed inset-0 z-[10000] flex items-end justify-center sm:items-center p-4 bg-black/50 backdrop-blur-sm transition-opacity duration-300">
    <div class="bg-white rounded-xl shadow-2xl p-6 max-w-md w-full transform transition-all scale-100">
      <div class="flex items-start space-x-4">
        <div class="text-4xl">🍪</div>
        <div class="flex-1">
          <h3 class="text-lg font-bold text-gray-900 mb-2">We Value Your Privacy</h3>
          <p class="text-sm text-gray-600 mb-4">
            We use cookies to enhance your browsing experience, serve personalized content, and analyze our traffic. By clicking 'Accept All', you consent to our use of cookies.
          </p>
          <div class="flex space-x-3">
            <button
              id="cookie-accept"
              @click="acceptCookies"
              class="flex-1 bg-[#005DAA] text-white px-4 py-2 rounded-lg font-semibold hover:bg-[#004a87] transition-colors shadow-sm"
            >
              Accept All
            </button>
            <button
              id="cookie-decline"
              @click="declineCookies"
              class="flex-1 bg-white border border-gray-300 text-gray-700 px-4 py-2 rounded-lg font-semibold hover:bg-gray-50 transition-colors"
            >
              Decline
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
    
    // Logic: Show if current page is HOME and consent not given
    const isVisible = computed(() => {
      return store.currentPageId === 'HOME' && store.cookie_consent_given !== true
    })

    const acceptCookies = () => {
      // FSM Action: ACT_HOME_ACCEPT_COOKIES
      // Effects: set cookie_consent_given = true
      store.cookie_consent_given = true
    }

    const declineCookies = () => {
      // Optional implementation
      store.cookie_consent_given = false
    }

    return {
      isVisible,
      acceptCookies,
      declineCookies
    }
  }
}
</script>