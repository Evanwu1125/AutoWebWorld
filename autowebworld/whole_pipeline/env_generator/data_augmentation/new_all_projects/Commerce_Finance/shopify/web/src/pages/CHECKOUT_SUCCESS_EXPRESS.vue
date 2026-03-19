<template>
  <div class="min-h-screen bg-white flex items-center justify-center p-4 text-center">
    <div class="max-w-md w-full">
         <div class="w-16 h-16 bg-[#5A31F4] rounded-full flex items-center justify-center mx-auto mb-6">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
         </div>
         <h1 class="text-2xl font-bold text-gray-900 mb-4">Express Payment Complete!</h1>
         <p class="text-gray-600 mb-8">Order #{{ signatureStore.order_id }} confirmed via {{ signatureStore.express_provider }}.</p>
         
         <button 
            id="success-express-go-home" 
            @click="goHome"
            class="w-full bg-[#008060] hover:bg-[#004C3F] text-white font-bold py-3 px-6 rounded-lg transition-colors"
        >
            Continue Shopping
        </button>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'CHECKOUT_SUCCESS_EXPRESS',
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