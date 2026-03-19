<template>
  <div class="min-h-screen bg-gray-50 flex flex-col items-center justify-center p-4 text-center">
    <div class="w-24 h-24 bg-green-100 rounded-full flex items-center justify-center mb-6 animate-bounce">
      <svg class="w-12 h-12 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
    </div>
    
    <h1 class="text-3xl font-black text-gray-900 mb-2">Order Placed!</h1>
    <p class="text-gray-600 mb-8">Your order is being processed and will be shipped soon.</p>
    
    <div class="bg-white p-6 rounded-xl shadow-sm w-full max-w-sm mb-8">
       <div class="flex justify-between mb-2">
          <span class="text-gray-500">Order ID</span>
          <span class="font-mono font-medium">{{ signatureStore.order_id || 'ORD-987654' }}</span>
       </div>
       <div class="flex justify-between mb-2">
          <span class="text-gray-500">Total</span>
          <span class="font-bold text-gray-900">$45.00</span>
       </div>
    </div>

    <button 
      id="checkout-buynow-success-go-home" 
      class="bg-red-600 text-white font-bold py-3 px-8 rounded-full shadow-lg hover:bg-red-700 transition-colors"
      @click="handleGoHome"
    >
      Back to Home
    </button>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'CHECKOUT_BUYNOW_SUCCESS',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const handleGoHome = async () => {
       signatureStore.currentPageId = 'HOME'
       await router.push({ name: 'HOME' })
    }

    return {
       signatureStore,
       handleGoHome
    }
  }
}
</script>