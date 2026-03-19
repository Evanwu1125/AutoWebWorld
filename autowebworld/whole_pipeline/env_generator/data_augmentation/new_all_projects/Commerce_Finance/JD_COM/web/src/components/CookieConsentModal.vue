<template>
  <div v-if="show" class="fixed inset-0 z-[10000] flex items-end justify-center sm:items-center p-4 bg-black/50 backdrop-blur-sm transition-opacity">
    <div class="bg-white rounded-xl shadow-2xl max-w-md w-full overflow-hidden animate-fade-in-up">
      <div class="p-6">
        <div class="flex items-center gap-3 mb-4">
          <div class="text-4xl">🍪</div>
          <h3 class="text-xl font-bold text-gray-900">We Value Your Privacy</h3>
        </div>
        <p class="text-gray-600 mb-6 leading-relaxed">
          We use cookies to enhance your browsing experience, serve personalized content, and analyze our traffic. By clicking "Accept All", you consent to our use of cookies.
        </p>
        <div class="flex gap-3">
          <button 
            id="cookie-accept"
            @click="accept"
            class="flex-1 bg-[#E1251B] hover:bg-[#c91f16] text-white font-semibold py-3 px-4 rounded-lg transition-colors duration-200 shadow-lg shadow-red-200"
          >
            Accept All
          </button>
          <!-- Secondary button not strictly required by FSM but good for UI -->
          <button 
            class="flex-1 bg-gray-100 hover:bg-gray-200 text-gray-700 font-semibold py-3 px-4 rounded-lg transition-colors duration-200"
            @click="accept" 
          >
            Close
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted, computed } from 'vue';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'CookieConsentModal',
  setup() {
    const signatureStore = useSignatureStore();
    
    const show = computed(() => {
      // Show if on HOME and cookie not accepted yet
      return signatureStore.currentPageId === 'HOME' && signatureStore.home_cookie_accepted !== true;
    });

    const accept = () => {
      // FSM Effect: set home_cookie_accepted = true
      signatureStore.home_cookie_accepted = true;
    };

    return {
      show,
      accept
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