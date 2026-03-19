<template>
  <div v-if="isVisible" class="fixed inset-0 z-[10000] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
    <div class="bg-[#272727] text-white rounded-xl shadow-2xl max-w-md w-full overflow-hidden border border-gray-700">
      <div class="p-6">
        <div class="flex items-center gap-3 mb-4">
          <div class="text-4xl">🍪</div>
          <h2 class="text-xl font-bold">We Value Your Privacy</h2>
        </div>
        <p class="text-gray-300 mb-6 leading-relaxed">
          We use cookies to enhance your browsing experience, serve personalized content, and analyze our traffic. By clicking "Accept All", you consent to our use of cookies.
        </p>
        <div class="flex flex-col gap-3">
          <button 
            id="cookie-accept"
            @click="acceptCookies"
            class="w-full bg-[#FF0000] hover:bg-red-600 text-white font-medium py-3 px-4 rounded-full transition-colors duration-200"
          >
            Accept All
          </button>
          <!-- Optional decline button for visual balance, though FSM only uses accept -->
          <button 
            class="w-full bg-transparent hover:bg-white/10 text-gray-300 font-medium py-3 px-4 rounded-full transition-colors duration-200"
            @click="isVisible = false" 
          >
            Customize
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
    
    // Show if consent is null (not yet given)
    const isVisible = computed(() => store.cookie_consent_given === null)
    
    const acceptCookies = () => {
      store.cookie_consent_given = true
    }
    
    return {
      isVisible,
      acceptCookies
    }
  }
}
</script>