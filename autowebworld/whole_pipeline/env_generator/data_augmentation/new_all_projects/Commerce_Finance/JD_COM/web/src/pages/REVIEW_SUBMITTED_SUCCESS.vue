<template>
  <div class="min-h-screen bg-[#F6F6F6] flex items-center justify-center p-4">
    <div class="bg-white rounded-2xl shadow-xl p-8 max-w-md w-full text-center">
      <div class="text-6xl mb-6">⭐</div>
      <h1 class="text-2xl font-bold text-gray-900 mb-2">Thanks for sharing!</h1>
      <p class="text-gray-600 mb-8">Your review helps others make better choices.</p>
      
      <button 
        id="go-home"
        @click="goHome"
        class="w-full bg-[#E1251B] hover:bg-[#c91f16] text-white font-bold py-3 px-4 rounded-xl shadow-lg shadow-red-200 transition-transform hover:scale-105"
      >
        Back to Home
      </button>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'REVIEW_SUBMITTED_SUCCESS',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();

    const goHome = async () => {
      signatureStore.currentPageId = 'HOME';
      await router.push({ name: 'HOME' });
    };

    return {
      goHome
    };
  }
}
</script>