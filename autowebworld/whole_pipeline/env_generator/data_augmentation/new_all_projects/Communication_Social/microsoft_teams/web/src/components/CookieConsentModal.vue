<template>
  <div v-if="show" class="fixed inset-0 z-[10000] flex items-end sm:items-center justify-center bg-black/50 backdrop-blur-sm p-4">
    <div class="bg-white rounded-xl shadow-2xl p-6 w-full max-w-lg transform transition-all">
      <div class="flex items-start gap-4">
        <div class="text-4xl">🍪</div>
        <div class="flex-1">
          <h3 class="text-xl font-bold text-gray-900 mb-2">We Value Your Privacy</h3>
          <p class="text-gray-600 mb-6 text-sm leading-relaxed">
            We use cookies to enhance your browsing experience, serve personalized content, and analyze our traffic. 
            By clicking "Accept All", you consent to our use of cookies as outlined in our Privacy Policy.
          </p>
          <div class="flex flex-col sm:flex-row gap-3">
            <button 
              id="cookie-accept"
              @click="accept"
              class="flex-1 bg-[#6264A7] hover:bg-[#464775] text-white font-semibold py-2.5 px-4 rounded-lg transition-all shadow-md active:transform active:scale-95"
            >
              Accept All
            </button>
            <button 
              class="flex-1 bg-gray-100 hover:bg-gray-200 text-gray-700 font-semibold py-2.5 px-4 rounded-lg transition-all"
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
import { useRoute } from 'vue-router'

export default {
  name: 'CookieConsentModal',
  setup() {
    const store = useSignatureStore()
    const route = useRoute()

    const show = computed(() => {
      // Only show on HOME
      if (route.name !== 'HOME') return false;
      
      // Check if consent is not yet given
      return store.cookie_consent_given !== true;
    })

    const accept = () => {
      // FSM Action: ACT_HOME_ACCEPT_COOKIES
      // Effect: set cookie_consent_given = true
      store.cookie_consent_given = true;
    }

    return {
      show,
      accept
    }
  }
}
</script>