<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <header class="bg-white shadow-sm px-4 py-3 flex items-center">
      <button 
        id="card-back-payments" 
        class="p-2 -ml-2 hover:bg-gray-100 rounded-full transition-colors"
        @click="handleBackPayments"
      >
        <svg class="w-6 h-6 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
      </button>
      <h1 class="text-lg font-bold text-gray-900 ml-2">Add Card</h1>
    </header>

    <div class="p-6 max-w-md mx-auto w-full space-y-6">
      <!-- Card Preview -->
      <div class="bg-gradient-to-r from-gray-800 to-gray-900 rounded-xl p-6 text-white shadow-lg h-48 flex flex-col justify-between relative overflow-hidden">
         <div class="absolute top-0 right-0 -mt-4 -mr-4 w-24 h-24 bg-white/10 rounded-full blur-xl"></div>
         <div class="flex justify-between items-start">
            <div class="text-xs opacity-75">Current Balance</div>
            <span class="text-2xl">💳</span>
         </div>
         <div>
            <div class="text-xl tracking-widest font-mono mb-1">{{ signatureStore.card_number || '**** **** **** ****' }}</div>
            <div class="flex justify-between items-end">
               <div>
                  <div class="text-xs opacity-75 uppercase">Card Holder</div>
                  <div class="font-medium tracking-wide">{{ signatureStore.card_holder || 'YOUR NAME' }}</div>
               </div>
               <div>
                  <div class="text-xs opacity-75 uppercase">Expires</div>
                  <div class="font-medium">{{ signatureStore.card_expiry || 'MM/YY' }}</div>
               </div>
            </div>
         </div>
      </div>

      <!-- Form -->
      <div class="bg-white p-6 rounded-xl shadow-sm space-y-4">
         <div>
            <label class="block text-xs font-bold text-gray-500 uppercase mb-1">Card Number</label>
            <input 
              id="card-number-input"
              type="text" 
              class="w-full border-b-2 border-gray-200 py-2 focus:border-red-500 focus:outline-none transition-colors"
              placeholder="0000 0000 0000 0000"
              :value="signatureStore.card_number"
              @input="e => signatureStore.card_number = e.target.value"
            />
         </div>

         <div>
            <label class="block text-xs font-bold text-gray-500 uppercase mb-1">Card Holder</label>
            <input 
              id="card-holder-input"
              type="text" 
              class="w-full border-b-2 border-gray-200 py-2 focus:border-red-500 focus:outline-none transition-colors"
              placeholder="NAME ON CARD"
              :value="signatureStore.card_holder"
              @input="e => signatureStore.card_holder = e.target.value"
            />
         </div>

         <div class="flex space-x-4">
            <div class="flex-1">
               <label class="block text-xs font-bold text-gray-500 uppercase mb-1">Expiry Date</label>
               <input 
                 id="card-expiry-input"
                 type="text" 
                 class="w-full border-b-2 border-gray-200 py-2 focus:border-red-500 focus:outline-none transition-colors"
                 placeholder="MM/YY"
                 :value="signatureStore.card_expiry"
                 @input="e => signatureStore.card_expiry = e.target.value"
               />
            </div>
            <div class="flex-1">
               <label class="block text-xs font-bold text-gray-500 uppercase mb-1">CVV</label>
               <input 
                 id="card-cvv-input"
                 type="password" 
                 class="w-full border-b-2 border-gray-200 py-2 focus:border-red-500 focus:outline-none transition-colors"
                 placeholder="123"
                 :value="signatureStore.card_cvv"
                 @input="e => signatureStore.card_cvv = e.target.value"
               />
            </div>
         </div>

         <button 
           id="card-authorize-button"
           class="w-full bg-red-600 text-white font-bold py-3 rounded-lg shadow-md hover:bg-red-700 transition-colors mt-4 disabled:opacity-50"
           :disabled="!canAuthorize"
           @click="handleAuthorize"
         >
           Save Card
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
  name: 'CARD_PAYMENT_GATEWAY',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const canAuthorize = computed(() => {
       const s = signatureStore
       return s.card_number && s.card_holder && s.card_expiry && s.card_cvv
    })

    const handleAuthorize = async () => {
       signatureStore.order_id = `CARD-${Date.now()}`
       signatureStore.success_message = 'Card Added Successfully'
       signatureStore.currentPageId = 'ORDER_CARD_SUCCESS'
       await router.push({ name: 'ORDER_CARD_SUCCESS' })
    }

    const handleBackPayments = async () => {
       signatureStore.currentPageId = 'PAYMENT_METHODS'
       await router.push({ name: 'PAYMENT_METHODS' })
    }

    return {
       signatureStore,
       canAuthorize,
       handleAuthorize,
       handleBackPayments
    }
  }
}
</script>