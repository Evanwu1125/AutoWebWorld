<template>
  <div v-if="visible" class="fixed inset-0 z-[10000] flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm transition-opacity">
    <div class="bg-white rounded-xl shadow-2xl max-w-md w-full overflow-hidden transform transition-all scale-100">
      <div class="p-6">
        <div class="flex items-center space-x-3 mb-4">
          <div class="bg-blue-100 p-2 rounded-full">
            <span class="text-2xl">🍪</span>
          </div>
          <h2 class="text-xl font-bold text-slate-800">We Value Your Privacy</h2>
        </div>
        <p class="text-slate-600 mb-6 leading-relaxed">
          We use cookies to enhance your secure messaging experience, remember your preferences, and ensure the application functions correctly. Your data stays private.
        </p>
        <div class="flex space-x-3">
          <button 
            id="cookie-accept"
            @click="accept"
            class="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-4 rounded-lg transition-colors shadow-md active:transform active:scale-95"
          >
            Accept All
          </button>
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
    
    // Visible if user is on HOME page and hasn't given consent yet
    const visible = computed(() => {
      return store.currentPageId === 'HOME' && store.cookie_consent_given === null
    })

    const accept = () => {
      // Update store to mark cookie consent as given
      store.cookie_consent_given = true
    }

    return {
      visible,
      accept,
      store
    }
  }
}
</script>