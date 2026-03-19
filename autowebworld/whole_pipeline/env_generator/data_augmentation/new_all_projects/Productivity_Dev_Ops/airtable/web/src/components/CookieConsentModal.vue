<template>
  <div v-if="isVisible" class="fixed inset-0 z-[10000] flex items-end sm:items-center justify-center p-4 bg-black/50 backdrop-blur-sm transition-opacity duration-300">
    <div class="bg-white rounded-xl shadow-2xl max-w-md w-full p-6 transform transition-all duration-300 scale-100">
      <div class="flex items-start gap-4">
        <div class="text-4xl">🍪</div>
        <div class="flex-1">
          <h2 class="text-xl font-bold text-gray-900 mb-2">We Value Your Privacy</h2>
          <p class="text-gray-600 text-sm mb-6 leading-relaxed">
            We use cookies to enhance your browsing experience, serve personalized content, and analyze our traffic. By clicking 'Accept All', you consent to our use of cookies.
          </p>
          <div class="flex gap-3">
            <button 
              id="cookie-accept"
              @click="handleAccept"
              class="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2.5 px-4 rounded-lg transition-colors focus:ring-4 focus:ring-blue-200"
            >
              Accept All
            </button>
            <button 
              class="flex-1 bg-gray-100 hover:bg-gray-200 text-gray-700 font-semibold py-2.5 px-4 rounded-lg transition-colors"
            >
              Customize
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
import { useRouter } from 'vue-router'

export default {
  name: 'CookieConsentModal',
  setup() {
    const store = useSignatureStore()
    const router = useRouter()
    
    // Show only on HOME page if consent not given
    const isVisible = computed(() => {
      return store.currentPageId === 'HOME' && store.cookie_consent_given !== true
    })

    const handleAccept = () => {
      // Logic maps to ACT_HOME_ACCEPT_COOKIES
      store.cookie_consent_given = true
    }

    return {
      isVisible,
      handleAccept
    }
  }
}
</script>