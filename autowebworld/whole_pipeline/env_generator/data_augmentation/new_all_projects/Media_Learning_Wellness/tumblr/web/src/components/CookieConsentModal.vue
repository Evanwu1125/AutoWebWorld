<template>
  <div v-if="isOpen" class="fixed inset-0 z-[10000] flex items-end sm:items-center justify-center p-4 bg-black/60 backdrop-blur-sm transition-opacity duration-300">
    <div class="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl max-w-md w-full p-6 transform transition-all duration-300 ease-out translate-y-0 opacity-100">
      <div class="flex items-start gap-4">
        <div class="text-3xl">🍪</div>
        <div class="flex-1">
          <h3 class="text-xl font-bold text-white mb-2">We Value Your Privacy</h3>
          <p class="text-slate-300 text-sm leading-relaxed mb-6">
            We use cookies to enhance your browsing experience, serve personalized content, and analyze our traffic. By clicking "Accept All", you consent to our use of cookies.
          </p>
          <div class="flex flex-col sm:flex-row gap-3">
            <button 
              id="cookie-accept"
              @click="accept"
              class="flex-1 bg-blue-500 hover:bg-blue-600 text-white font-bold py-2.5 px-4 rounded-full transition-colors duration-200 text-sm"
            >
              Accept All
            </button>
            <button 
              @click="decline"
              class="flex-1 bg-transparent hover:bg-slate-800 text-slate-400 font-semibold py-2.5 px-4 rounded-full border border-slate-700 transition-colors duration-200 text-sm"
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
    
    // Only show if consent is null (not yet decided)
    const isOpen = computed(() => store.cookie_consent_given === null)

    const accept = () => {
      // FSM Effect: set cookie_consent_given = true
      store.cookie_consent_given = true
    }

    const decline = () => {
      // Not strictly in FSM happy path, but good UX
      store.cookie_consent_given = false
    }

    return {
      isOpen,
      accept,
      decline
    }
  }
}
</script>