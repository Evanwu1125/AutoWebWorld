<template>
  <div class="min-h-screen bg-white flex items-center justify-center p-4">
    <div class="text-center max-w-md">
      <div class="w-24 h-24 bg-green-100 text-green-500 rounded-full flex items-center justify-center mx-auto mb-6 animate-bounce">
        <svg class="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
      </div>
      <h1 class="text-3xl font-bold text-gray-900 mb-4">Welcome Aboard!</h1>
      <p class="text-gray-600 mb-8 text-lg">Your account has been successfully created. You can now enjoy member benefits.</p>
      
      <button 
        id="go-home"
        @click="goHome"
        class="bg-[#E1251B] hover:bg-[#c91f16] text-white font-bold py-3 px-8 rounded-full shadow-lg shadow-red-200 transition-transform hover:scale-105"
      >
        Start Shopping
      </button>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'ACCOUNT_CREATED_SUCCESS',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();

    const goHome = async () => {
      // Auto login effect
      signatureStore.current_user_id = 'new_user';
      signatureStore.currentPageId = 'HOME';
      await router.push({ name: 'HOME' });
    };

    return {
      goHome
    };
  }
}
</script>