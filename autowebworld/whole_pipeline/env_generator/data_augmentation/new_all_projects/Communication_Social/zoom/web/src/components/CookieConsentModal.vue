<template>
  <div v-if="show" class="fixed inset-0 z-[10000] flex items-end sm:items-center justify-center bg-black/30 backdrop-blur-[2px] p-4">
    <div class="bg-white dark:bg-gray-800 rounded-xl shadow-2xl max-w-lg w-full p-6 transform transition-all animate-fade-in-up">
      <div class="flex flex-col sm:flex-row gap-4 items-start">
        <div class="flex-shrink-0 bg-blue-100 dark:bg-blue-900/30 p-3 rounded-full">
          <span class="text-2xl">🍪</span>
        </div>
        
        <div class="flex-1">
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-2">We Value Your Privacy</h3>
          <p class="text-sm text-gray-600 dark:text-gray-300 mb-4 leading-relaxed">
            We use cookies to enhance your browsing experience, serve personalized content, and analyze our traffic. By clicking "Accept All", you consent to our use of cookies.
          </p>
          
          <div class="flex flex-col sm:flex-row gap-3">
            <button 
              id="cookie-accept"
              @click="acceptCookies"
              class="flex-1 py-2.5 px-4 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 shadow-sm"
            >
              Accept All
            </button>
            <button 
              id="cookie-decline"
              @click="declineCookies"
              class="flex-1 py-2.5 px-4 bg-white border border-gray-300 hover:bg-gray-50 text-gray-700 font-medium rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 dark:bg-gray-700 dark:border-gray-600 dark:text-gray-200 dark:hover:bg-gray-600"
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
import { computed } from 'vue';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'CookieConsentModal',
  setup() {
    const store = useSignatureStore();
    
    const show = computed(() => {
      // Only show on HOME page and if not yet accepted
      return store.currentPageId === 'HOME' && store.cookie_accepted !== true;
    });

    const acceptCookies = () => {
      // Invoke FSM action
      store.handleAction('ACT_HOME_ACCEPT_COOKIE');
      // Fallback if action logic misses
      if (store.cookie_accepted !== true) {
        store.cookie_accepted = true;
      }
    };

    const declineCookies = () => {
      // Optional logic
    };

    return {
      show,
      acceptCookies,
      declineCookies
    };
  }
}
</script>

<style scoped>
@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.animate-fade-in-up {
  animation: fadeInUp 0.4s ease-out forwards;
}
</style>