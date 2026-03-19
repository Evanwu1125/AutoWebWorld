<template>
  <div v-if="isVisible" class="fixed inset-0 z-[10000] flex items-end sm:items-center justify-center bg-black/50 backdrop-blur-sm p-4 transition-opacity duration-300">
    <div class="bg-white rounded-xl shadow-2xl max-w-md w-full p-6 transform transition-all scale-100">
      <div class="flex items-start gap-4">
        <div class="text-4xl">🍪</div>
        <div>
          <h3 class="text-lg font-bold text-gray-900 mb-2">We Value Your Privacy</h3>
          <p class="text-sm text-gray-600 mb-4">
            We use cookies to enhance your browsing experience, serve personalized content, and analyze our traffic. By clicking 'Accept All', you consent to our use of cookies.
          </p>
        </div>
      </div>
      <div class="flex justify-end gap-3 mt-2">
        <button 
          id="cookie-decline"
          @click="decline"
          class="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
        >
          Decline
        </button>
        <button 
          id="cookie-accept"
          @click="accept"
          class="px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-lg shadow-sm transition-colors"
        >
          Accept All
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'CookieConsentModal',
  setup() {
    const signatureStore = useSignatureStore()
    
    // Only show if consent is null (not yet given or declined)
    // And we are on HOME page (as per FSM logic usually, but here global check is fine or check route)
    // FSM says ACT_HOME_ACCEPT_COOKIES is on HOME page. 
    const isVisible = computed(() => {
      return signatureStore.currentPageId === 'HOME' && signatureStore.cookie_consent_given !== true
    })

    function accept() {
      signatureStore.cookie_consent_given = true
    }

    function decline() {
      // Optional: handle decline
      signatureStore.cookie_consent_given = false
    }

    return {
      isVisible,
      accept,
      decline
    }
  }
}
</script>