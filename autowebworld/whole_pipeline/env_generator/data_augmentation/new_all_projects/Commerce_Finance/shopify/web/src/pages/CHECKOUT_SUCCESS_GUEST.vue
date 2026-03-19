<template>
  <div class="min-h-screen bg-white flex items-center justify-center p-4 text-center">
    <div class="max-w-md w-full">
         <div class="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-6">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8 text-[#008060]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
            </svg>
         </div>
         <h1 class="text-2xl font-bold text-gray-900 mb-4">Order Placed!</h1>
         <p class="text-gray-600 mb-8">Thank you for ordering as a guest. Order #{{ signatureStore.order_id }}</p>
         
         <button 
            id="success-guest-go-home" 
            @click="goHome"
            class="w-full bg-[#008060] hover:bg-[#004C3F] text-white font-bold py-3 px-6 rounded-lg transition-colors"
        >
            Back to Home
        </button>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'CHECKOUT_SUCCESS_GUEST',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const goHome = async () => {
        signatureStore.cart_items = []
        signatureStore.currentPageId = 'HOME'
        await router.push({ name: 'HOME' })
    }

    return {
        signatureStore,
        goHome
    }
  }
}
</script>