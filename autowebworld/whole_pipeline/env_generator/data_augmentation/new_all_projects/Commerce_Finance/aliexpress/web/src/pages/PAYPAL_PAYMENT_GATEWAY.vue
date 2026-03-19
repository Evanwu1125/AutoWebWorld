<template>
  <div class="min-h-screen bg-white flex flex-col">
    <!-- Mock Paypal Header -->
    <header class="bg-white border-b border-gray-200 px-4 py-4 flex justify-center relative">
      <button 
        id="paypal-back-payments" 
        class="absolute left-4 text-gray-500 hover:text-gray-700"
        @click="handleBackPayments"
      >
        Cancel
      </button>
      <span class="font-bold text-blue-900 italic text-xl">PayPal</span>
    </header>

    <div class="flex-1 flex flex-col items-center justify-center p-6 max-w-md mx-auto w-full">
      <h2 class="text-2xl font-light text-gray-700 mb-8">Log in to PayPal</h2>
      
      <div class="w-full space-y-4">
         <input 
           id="paypal-email-input"
           type="email" 
           placeholder="Email or mobile number" 
           class="w-full px-4 py-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:outline-none"
           :value="signatureStore.paypal_email"
           @input="handleEmailInput"
         />
         
         <button 
           id="paypal-authorize-button"
           class="w-full bg-blue-800 text-white font-bold py-3 rounded-md hover:bg-blue-900 transition-colors shadow-sm disabled:opacity-50"
           :disabled="!canAuthorize"
           @click="handleAuthorize"
         >
           Next
         </button>
         
         <div class="text-center mt-4">
            <a href="#" class="text-sm text-blue-600 hover:underline">Forgot password?</a>
         </div>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'PAYPAL_PAYMENT_GATEWAY',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const canAuthorize = computed(() => {
       return signatureStore.paypal_email && signatureStore.paypal_email.length > 0
    })

    const handleEmailInput = (e) => {
       signatureStore.paypal_email = e.target.value
    }

    const handleAuthorize = async () => {
       // Mock authorization
       signatureStore.order_id = `PAYPAL-${Date.now()}`
       signatureStore.success_message = 'PayPal Connected Successfully'
       signatureStore.currentPageId = 'ORDER_PAYPAL_SUCCESS'
       await router.push({ name: 'ORDER_PAYPAL_SUCCESS' })
    }

    const handleBackPayments = async () => {
       signatureStore.currentPageId = 'PAYMENT_METHODS'
       await router.push({ name: 'PAYMENT_METHODS' })
    }

    return {
       signatureStore,
       canAuthorize,
       handleEmailInput,
       handleAuthorize,
       handleBackPayments
    }
  }
}
</script>