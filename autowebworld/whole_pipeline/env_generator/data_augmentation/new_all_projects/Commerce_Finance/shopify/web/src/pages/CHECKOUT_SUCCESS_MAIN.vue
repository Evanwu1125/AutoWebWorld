<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center p-4">
    <div class="bg-white p-8 rounded-xl shadow-lg max-w-md w-full text-center">
      <div class="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-6">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8 text-[#008060]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
        </svg>
      </div>
      <h1 class="text-2xl font-bold text-gray-900 mb-2">Order Confirmed!</h1>
      <p class="text-gray-600 mb-8">
        Thank you for your purchase. Your order <span class="font-bold text-gray-900">{{ orderId }}</span> has been received.
      </p>
      
      <div class="space-y-3">
        <button 
          id="success-main-view-order" 
          @click="viewOrder" 
          class="w-full bg-[#008060] hover:bg-[#004C3F] text-white font-bold py-3 px-6 rounded-lg transition-colors"
        >
          View Order Details
        </button>
        <button 
          id="success-main-go-home" 
          @click="goHome" 
          class="w-full bg-white border border-gray-300 hover:bg-gray-50 text-gray-700 font-bold py-3 px-6 rounded-lg transition-colors"
        >
          Back to Home
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'CHECKOUT_SUCCESS_MAIN',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    
    const orderId = computed(() => signatureStore.order_id || '#12345')

    const viewOrder = async () => {
        signatureStore.currentPageId = 'ORDER_CONFIRMATION_SUCCESS'
        await router.push({ name: 'ORDER_CONFIRMATION_SUCCESS' })
    }

    const goHome = async () => {
        // Clear cart on success navigation if desired, or let FSM effects handle it (here simplified)
        signatureStore.cart_items = []
        signatureStore.cart_subtotal = 0
        signatureStore.currentPageId = 'HOME'
        await router.push({ name: 'HOME' })
    }

    return {
        orderId,
        viewOrder,
        goHome
    }
  }
}
</script>