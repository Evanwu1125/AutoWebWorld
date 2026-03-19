<template>
  <transition name="fade">
    <div v-if="isVisible" class="fixed inset-0 z-[10000] flex items-end sm:items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
      <div class="bg-white rounded-xl shadow-2xl max-w-md w-full p-6 transform transition-all">
        <div class="flex items-start gap-4">
          <div class="text-4xl">🍪</div>
          <div class="flex-1">
            <h3 class="text-lg font-bold text-gray-900 mb-2">We Value Your Privacy</h3>
            <p class="text-gray-600 text-sm mb-6">
              We use cookies to enhance your browsing experience, serve personalized content, and analyze our traffic. By clicking "Accept All", you consent to our use of cookies.
            </p>
            <div class="flex gap-3 justify-end">
              <button 
                @click="decline"
                class="px-4 py-2 text-sm font-medium text-gray-600 bg-gray-100 rounded-full hover:bg-gray-200 transition-colors"
              >
                Decline
              </button>
              <button 
                id="cookie-accept"
                @click="accept"
                class="px-6 py-2 text-sm font-medium text-white bg-blue-600 rounded-full hover:bg-blue-700 shadow-md hover:shadow-lg transition-all"
              >
                Accept All
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </transition>
</template>

<script>
import { computed } from 'vue'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'CookieConsentModal',
  setup() {
    const store = useSignatureStore()
    
    const isVisible = computed(() => {
      // Show only on HOME page if consent is null
      return store.currentPageId === 'HOME' && store.cookie_consent_given === null
    })

    const accept = () => {
      // FSM: ACT_HOME_ACCEPT_COOKIES
      // Effect: Set cookie_consent_given to true
      store.cookie_consent_given = true
    }

    const decline = () => {
      // Set cookie_consent_given to false
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
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>