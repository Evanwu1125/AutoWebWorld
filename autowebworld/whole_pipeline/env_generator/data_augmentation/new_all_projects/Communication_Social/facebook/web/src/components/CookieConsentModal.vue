<template>
  <div v-if="isVisible" class="fixed inset-0 z-[10000] flex items-end sm:items-center justify-center p-4 bg-black/50 backdrop-blur-sm transition-opacity duration-300">
    <div class="bg-white rounded-xl shadow-2xl w-full max-w-lg overflow-hidden transform transition-all duration-300 scale-100">
      <div class="p-6">
        <div class="flex items-start gap-4">
          <div class="flex-shrink-0 bg-blue-100 p-3 rounded-full">
            <span class="text-2xl">🍪</span>
          </div>
          <div class="flex-1">
            <h3 class="text-lg font-semibold text-gray-900 mb-2">We Value Your Privacy</h3>
            <p class="text-gray-600 text-sm leading-relaxed">
              We use cookies to enhance your browsing experience, serve personalized content, and analyze our traffic. By clicking 'Accept All', you consent to our use of cookies.
            </p>
          </div>
        </div>
        <div class="mt-6 flex justify-end gap-3">
          <button class="px-4 py-2 text-sm font-medium text-gray-600 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors">
            Customize
          </button>
          <button 
            id="cookie-accept"
            @click="acceptCookies"
            class="px-6 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-lg shadow-sm transition-colors"
          >
            Accept All
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'CookieConsentModal',
  setup() {
    const signatureStore = useSignatureStore();
    
    const isVisible = computed(() => {
      // Show only on HOME page and if consent not given
      return signatureStore.currentPageId === 'HOME' && signatureStore.cookie_consent_given === null;
    });
    
    const acceptCookies = () => {
      // Matches FSM action HOME_ACCEPT_COOKIES
      signatureStore.cookie_consent_given = true;
    };
    
    return {
      isVisible,
      acceptCookies
    };
  }
}
</script>