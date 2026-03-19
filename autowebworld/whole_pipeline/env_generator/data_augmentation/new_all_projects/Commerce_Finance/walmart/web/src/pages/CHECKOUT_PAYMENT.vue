<template>
  <div class="checkout-page min-h-screen bg-gray-50 flex flex-col">
    <header class="bg-white border-b p-4 sticky top-0 z-30">
      <div class="max-w-3xl mx-auto flex justify-center">
         <div class="font-bold text-xl text-[#0071DC] flex items-center gap-2">
            <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" /></svg>
            Walmart Checkout
         </div>
      </div>
    </header>

    <main class="flex-1 max-w-3xl mx-auto w-full p-4 md:p-8">
      <div class="bg-white rounded-xl shadow-sm overflow-hidden">
        <div class="flex border-b">
           <div class="flex-1 py-3 text-center text-green-600 font-bold flex items-center justify-center gap-1">
             <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" /></svg>
             1. Shipping
           </div>
           <div class="flex-1 py-3 text-center text-blue-600 font-bold border-b-2 border-blue-600">2. Payment</div>
           <div class="flex-1 py-3 text-center text-gray-400 font-medium">3. Review</div>
        </div>

        <div class="p-6 md:p-8 space-y-6">
           <h2 class="text-2xl font-bold mb-6">Payment Method</h2>

           <div class="space-y-4">
             <!-- Payment Method Selector -->
             <div class="form-group">
                <label class="block text-sm font-medium text-gray-700 mb-2">Select Method</label>
                <div class="relative">
                   <button 
                     id="payment-method-dropdown"
                     @click="showMethodDropdown = !showMethodDropdown"
                     class="w-full flex items-center justify-between px-4 py-3 border border-gray-300 rounded-lg bg-white text-left hover:border-gray-400"
                   >
                     <span class="flex items-center gap-2">
                       <svg v-if="method === 'card'" class="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z" /></svg>
                       {{ method === 'card' ? 'Credit/Debit Card' : 'PayPal' }}
                     </span>
                     <svg class="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" /></svg>
                   </button>
                   
                   <div v-if="showMethodDropdown" class="absolute z-10 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg">
                      <div 
                        id="payment-method-card"
                        @click="selectMethod('card')"
                        class="px-4 py-3 hover:bg-gray-50 cursor-pointer flex justify-between items-center"
                      >
                         <div>Credit/Debit Card</div>
                         <span v-if="method === 'card'" class="text-blue-600 font-bold">✓</span>
                      </div>
                      <div 
                        id="payment-method-paypal"
                        @click="selectMethod('paypal')"
                        class="px-4 py-3 hover:bg-gray-50 cursor-pointer flex justify-between items-center border-t border-gray-100"
                      >
                         <div>PayPal</div>
                         <span v-if="method === 'paypal'" class="text-blue-600 font-bold">✓</span>
                      </div>
                   </div>
                </div>
             </div>

             <!-- Card Form -->
             <div v-if="method === 'card'" class="p-4 bg-gray-50 rounded-lg border border-gray-200 space-y-4">
                <div class="form-group">
                   <label class="block text-xs font-bold text-gray-500 uppercase tracking-wide mb-1">Card Number</label>
                   <input 
                     id="card-number-input"
                     type="text" 
                     v-model="cardNumber"
                     @input="updateCardNumber"
                     class="w-full px-4 py-2 border border-gray-300 rounded bg-white focus:ring-2 focus:ring-blue-500 outline-none"
                     placeholder="0000 0000 0000 0000"
                   />
                </div>
                <div class="grid grid-cols-2 gap-4">
                   <div class="form-group">
                      <label class="block text-xs font-bold text-gray-500 uppercase tracking-wide mb-1">Expiry Date</label>
                      <input 
                        id="card-expiry-input"
                        type="text" 
                        v-model="cardExpiry"
                        @input="updateCardExpiry"
                        class="w-full px-4 py-2 border border-gray-300 rounded bg-white focus:ring-2 focus:ring-blue-500 outline-none"
                        placeholder="MM/YY"
                      />
                   </div>
                   <div class="form-group">
                      <label class="block text-xs font-bold text-gray-500 uppercase tracking-wide mb-1">CVV</label>
                      <input 
                        id="card-cvv-input"
                        type="text" 
                        v-model="cardCVV"
                        @input="updateCardCVV"
                        class="w-full px-4 py-2 border border-gray-300 rounded bg-white focus:ring-2 focus:ring-blue-500 outline-none"
                        placeholder="123"
                      />
                   </div>
                </div>
             </div>
           </div>

           <!-- Actions -->
           <div class="pt-6 border-t flex items-center justify-between">
              <button 
                id="payment-back-to-shipping"
                @click="handleBackToShipping"
                class="text-gray-600 font-medium hover:text-[#0071DC] hover:underline"
              >
                &larr; Back to Shipping
              </button>
              <button 
                id="payment-continue-button"
                @click="handleContinue"
                :disabled="!isFormValid"
                class="bg-[#0071DC] text-white font-bold py-3 px-8 rounded-full shadow-md hover:bg-[#005bb5] disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                Review Order
              </button>
           </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'CHECKOUT_PAYMENT',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const cardNumber = ref(store.card_number || '')
    const cardExpiry = ref(store.card_expiry || '')
    const cardCVV = ref(store.card_cvv || '')
    const method = ref(store.payment_method || 'card')
    const showMethodDropdown = ref(false)

    const isFormValid = computed(() => {
      if (method.value === 'paypal') return true // Mock valid for paypal
      return cardNumber.value && cardExpiry.value && cardCVV.value
    })

    const updateCardNumber = () => { store.card_number = cardNumber.value } // FSM: ACT_PAYMENT_ENTER_CARD_NUMBER
    const updateCardExpiry = () => { store.card_expiry = cardExpiry.value } // FSM: ACT_PAYMENT_ENTER_EXPIRY
    const updateCardCVV = () => { store.card_cvv = cardCVV.value } // FSM: ACT_PAYMENT_ENTER_CVV
    
    const selectMethod = (val) => {
      // FSM: ACT_PAYMENT_SELECT_METHOD
      method.value = val
      store.payment_method = val
      showMethodDropdown.value = false
    }

    const handleContinue = async () => {
      // FSM: ACT_PAYMENT_CONTINUE_TO_REVIEW
      store.currentPageId = 'CHECKOUT_REVIEW'
      await router.push({ name: 'CHECKOUT_REVIEW' })
    }

    const handleBackToShipping = async () => {
      // FSM: ACT_PAYMENT_BACK_TO_SHIPPING
      store.currentPageId = 'CHECKOUT_SHIPPING'
      await router.push({ name: 'CHECKOUT_SHIPPING' })
    }

    return {
      cardNumber, cardExpiry, cardCVV, method, showMethodDropdown, isFormValid,
      updateCardNumber, updateCardExpiry, updateCardCVV, selectMethod,
      handleContinue, handleBackToShipping
    }
  }
}
</script>