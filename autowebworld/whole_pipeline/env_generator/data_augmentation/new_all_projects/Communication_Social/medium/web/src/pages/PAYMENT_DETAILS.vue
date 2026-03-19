<template>
  <div class="min-h-screen bg-white flex flex-col items-center pt-12 pb-12 px-4">
    <div class="w-full max-w-md">
       <button id="payment-back-membership" @click="handleBack" class="text-gray-500 hover:text-black mb-8 flex items-center gap-2 text-sm font-sans">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
          </svg>
          Change Plan
       </button>

       <h2 class="text-3xl font-bold font-serif mb-8 text-center">Payment Details</h2>
       
       <div class="bg-gray-50 p-6 rounded-xl border border-gray-200 shadow-sm space-y-6">
          <div>
             <label class="block text-xs font-bold uppercase text-gray-500 font-sans mb-2">Card Number</label>
             <input 
                id="payment-card-number"
                v-model="cardNumber"
                type="text" 
                placeholder="0000 0000 0000 0000"
                class="w-full p-3 rounded border border-gray-300 focus:border-green-500 focus:ring-1 focus:ring-green-500 font-mono text-lg"
             />
          </div>
          
          <div>
             <label class="block text-xs font-bold uppercase text-gray-500 font-sans mb-2">Cardholder Name</label>
             <input 
                id="payment-card-name"
                v-model="cardName"
                type="text" 
                placeholder="John Doe"
                class="w-full p-3 rounded border border-gray-300 focus:border-green-500 focus:ring-1 focus:ring-green-500 font-sans"
             />
          </div>
          
          <div class="flex gap-4">
             <div class="flex-1">
                <label class="block text-xs font-bold uppercase text-gray-500 font-sans mb-2">Expiry</label>
                <input 
                   type="text" 
                   placeholder="MM/YY"
                   class="w-full p-3 rounded border border-gray-300 focus:border-green-500 focus:ring-1 focus:ring-green-500 font-mono"
                />
             </div>
             <div class="w-24">
                <label class="block text-xs font-bold uppercase text-gray-500 font-sans mb-2">CVV</label>
                <input 
                   id="payment-card-cvv"
                   v-model="cardCvv"
                   type="text" 
                   placeholder="123"
                   class="w-full p-3 rounded border border-gray-300 focus:border-green-500 focus:ring-1 focus:ring-green-500 font-mono"
                />
             </div>
          </div>
       </div>

       <div class="mt-8 space-y-4">
          <div class="flex items-center gap-2 mb-4">
             <input 
                type="checkbox" 
                id="payment-enable-pay" 
                @click="handleEnablePay"
                :disabled="!formFilled"
                class="rounded text-green-600 focus:ring-green-500 border-gray-300" 
             />
             <label for="payment-enable-pay" class="text-sm text-gray-600 font-sans">I authorize this recurring payment.</label>
          </div>

          <button 
             id="payment-submit" 
             @click="handleSubmit" 
             :disabled="!paymentReady"
             :class="{
                'w-full py-4 rounded-full text-lg font-medium font-sans transition-all shadow-lg': true,
                'bg-black text-white hover:bg-gray-800': paymentReady,
                'bg-gray-200 text-gray-400 cursor-not-allowed shadow-none': !paymentReady
             }"
          >
             Pay Now
          </button>
       </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'PAYMENT_DETAILS',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    
    const cardNumber = ref('')
    const cardName = ref('')
    const cardCvv = ref('')
    
    const formFilled = computed(() => cardNumber.value.length > 0 && cardName.value.length > 0 && cardCvv.value.length > 0)
    const paymentReady = computed(() => signatureStore.payment_ready === true)

    watch(cardNumber, (val) => { if(val) signatureStore.card_number_entered = true })
    watch(cardName, (val) => { if(val) signatureStore.card_name_entered = true })
    watch(cardCvv, (val) => { if(val) signatureStore.card_cvv_entered = true })

    const handleEnablePay = () => {
       if (formFilled.value) {
          signatureStore.payment_ready = true
       }
    }

    const handleSubmit = async () => {
       if (paymentReady.value) {
          signatureStore.setCurrentPageId('SUBSCRIPTION_SUCCESS')
          await router.push({ name: 'SUBSCRIPTION_SUCCESS' })
       }
    }

    const handleBack = async () => {
       signatureStore.setCurrentPageId('MEMBERSHIP_LANDING')
       await router.push({ name: 'MEMBERSHIP_LANDING' })
    }

    return {
       cardNumber,
       cardName,
       cardCvv,
       formFilled,
       paymentReady,
       handleEnablePay,
       handleSubmit,
       handleBack
    }
  }
}
</script>