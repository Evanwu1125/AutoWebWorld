<template>
  <div v-if="isVisible" class="fixed inset-0 z-[10000] flex items-center justify-center bg-black/50 backdrop-blur-sm transition-opacity duration-300">
    <div class="bg-white rounded-xl shadow-2xl p-6 max-w-md w-full mx-4 transform transition-all scale-100">
      <div class="text-center">
        <div class="text-4xl mb-4">🍪</div>
        <h3 class="text-xl font-bold text-gray-900 mb-2">We Value Your Privacy</h3>
        <p class="text-gray-600 mb-6 text-sm leading-relaxed">
          We use cookies to enhance your browsing experience, serve personalized content, and analyze our traffic. By clicking 'Accept All', you consent to our use of cookies.
        </p>
        <button 
          id="cookie-accept"
          @click="acceptCookies"
          class="w-full bg-[#B92B27] hover:bg-[#a02521] text-white font-bold py-3 px-6 rounded-full transition-colors duration-200 shadow-md transform active:scale-95"
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
    
    // Show only on HOME page if consent not given
    const isVisible = computed(() => {
      return store.currentPageId === 'HOME' && store.cookie_consent_given !== true
    })

    function acceptCookies() {
      // Direct store update as per FSM effect, handled in HOME page action usually, 
      // but here we simulate the action trigger or directly update if FSM logic is tightly coupled in component
      // In FSM-based approach, the button click triggers the action handler in the page.
      // However, since this is a global modal component, we emit the event.
      // BUT, FSM says "gui_procedure" points to #cookie-accept.
      // So the Page component (HOME.vue) should handle the click event if possible, 
      // OR this component updates store directly if it encapsulates the logic.
      // Given FSM structure: ACT_ACCEPT_COOKIES is on HOME page.
      // We will emit an event so HOME.vue can handle it, OR we just let the click propagate if it was slot based.
      // Better: The click handler here updates the store state to close modal visually, 
      // but strictly speaking HOME.vue should attach the listener.
      // Since this is a Vue component, we'll let the parent handle it or simple update store here to close it.
      // FSM Requirement: ACT_ACCEPT_COOKIES effects: set cookie_consent_given = true.
      
      store.cookie_consent_given = true
    }

    return {
      isVisible,
      acceptCookies
    }
  }
}
</script>