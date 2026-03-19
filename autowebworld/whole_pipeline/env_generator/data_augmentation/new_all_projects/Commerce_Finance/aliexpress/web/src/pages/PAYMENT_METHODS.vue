<template>
  <div class="min-h-screen bg-gray-50 pb-20 font-sans">
    <header class="bg-white shadow-sm px-4 py-3 flex items-center sticky top-0 z-20">
      <button 
        id="payments-back-account" 
        class="p-2 -ml-2 hover:bg-gray-100 rounded-full transition-colors"
        @click="handleBackAccount"
      >
        <svg class="w-6 h-6 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
      </button>
      <h1 class="text-lg font-bold text-gray-900 ml-2">Payment Methods</h1>
    </header>

    <div class="p-4 space-y-4">
       <!-- Add Card Option -->
       <div 
         id="payment-method-card" 
         class="bg-white p-4 rounded-xl shadow-sm border-2 cursor-pointer transition-all flex items-center justify-between"
         :class="signatureStore.selected_payment_method === 'card' ? 'border-red-600 ring-1 ring-red-600' : 'border-transparent hover:border-gray-200'"
         @click="handleSelectMethod('card')"
       >
          <div class="flex items-center space-x-3">
             <span class="text-2xl">💳</span>
             <div>
                <h3 class="font-bold text-gray-900">Credit/Debit Card</h3>
                <p class="text-xs text-gray-500">Visa, Mastercard, etc.</p>
             </div>
          </div>
          <div v-if="signatureStore.selected_payment_method === 'card'" class="text-red-600">
             <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
          </div>
       </div>

       <!-- Add PayPal Option -->
       <div 
         id="payment-method-paypal" 
         class="bg-white p-4 rounded-xl shadow-sm border-2 cursor-pointer transition-all flex items-center justify-between"
         :class="signatureStore.selected_payment_method === 'paypal' ? 'border-blue-600 ring-1 ring-blue-600' : 'border-transparent hover:border-gray-200'"
         @click="handleSelectMethod('paypal')"
       >
          <div class="flex items-center space-x-3">
             <span class="text-2xl">🅿️</span>
             <div>
                <h3 class="font-bold text-gray-900">PayPal</h3>
                <p class="text-xs text-gray-500">Connect your PayPal account</p>
             </div>
          </div>
          <div v-if="signatureStore.selected_payment_method === 'paypal'" class="text-blue-600">
             <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
          </div>
       </div>
       
       <!-- Action Buttons based on selection -->
       <div v-if="signatureStore.selected_payment_method === 'card'" class="mt-6">
          <button 
            id="payment-save-card" 
            class="w-full bg-red-600 text-white font-bold py-3 rounded-lg shadow-md hover:bg-red-700 transition-colors"
            @click="handleGoCardGateway"
          >
            Add New Card
          </button>
       </div>

       <div v-if="signatureStore.selected_payment_method === 'paypal'" class="mt-6">
          <button 
            id="payment-save-paypal" 
            class="w-full bg-blue-600 text-white font-bold py-3 rounded-lg shadow-md hover:bg-blue-700 transition-colors"
            @click="handleGoPaypalGateway"
          >
            Connect PayPal
          </button>
       </div>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'PAYMENT_METHODS',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const handleBackAccount = async () => {
       signatureStore.currentPageId = 'ACCOUNT_OVERVIEW'
       await router.push({ name: 'ACCOUNT_OVERVIEW' })
    }

    const handleSelectMethod = (method) => {
       signatureStore.selected_payment_method = method
    }

    const handleGoCardGateway = async () => {
       signatureStore.currentPageId = 'CARD_PAYMENT_GATEWAY'
       await router.push({ name: 'CARD_PAYMENT_GATEWAY' })
    }

    const handleGoPaypalGateway = async () => {
       signatureStore.currentPageId = 'PAYPAL_PAYMENT_GATEWAY'
       await router.push({ name: 'PAYPAL_PAYMENT_GATEWAY' })
    }

    return {
       signatureStore,
       handleBackAccount,
       handleSelectMethod,
       handleGoCardGateway,
       handleGoPaypalGateway
    }
  }
}
</script>