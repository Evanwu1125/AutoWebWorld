<template>
  <div class="min-h-screen bg-[#F6F6F6] flex items-center justify-center p-4">
    <div class="bg-white rounded-2xl shadow-xl p-8 max-w-md w-full text-center">
      <div class="w-20 h-20 bg-green-100 text-green-500 rounded-full flex items-center justify-center mx-auto mb-6">
        <svg class="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
      </div>
      <h1 class="text-2xl font-bold text-gray-900 mb-2">Order Placed Successfully!</h1>
      <p class="text-gray-600 mb-8">Thank you for your purchase. Your order from cart has been received and is being processed.</p>
      
      <button 
        id="go-home"
        @click="goHome"
        class="w-full bg-[#E1251B] hover:bg-[#c91f16] text-white font-bold py-3 px-4 rounded-xl shadow-lg shadow-red-200 transition-transform hover:scale-105"
      >
        Continue Shopping
      </button>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'CHECKOUT_FROM_CART_SUCCESS',
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