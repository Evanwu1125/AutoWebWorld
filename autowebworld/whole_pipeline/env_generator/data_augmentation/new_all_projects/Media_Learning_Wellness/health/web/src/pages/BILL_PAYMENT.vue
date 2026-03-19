<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <header class="bg-white shadow-sm z-10">
       <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center">
         <button id="back-bill-detail" @click="handleBack" class="mr-4 text-gray-600 hover:text-gray-900">
           <svg class="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>
         </button>
         <h1 class="text-xl font-bold text-gray-900">Secure Payment</h1>
       </div>
    </header>

    <main class="flex-1 max-w-xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
      <div class="bg-white rounded-lg shadow-lg p-6">
        <div class="mb-6 border-b border-gray-100 pb-4">
           <p class="text-sm text-gray-500">Amount Due</p>
           <p class="text-3xl font-bold text-gray-900">${{ store.billing_amount_due.toFixed(2) }}</p>
        </div>

        <div class="space-y-6">
           <div>
              <label for="card-number-input" class="block text-sm font-medium text-gray-700 mb-2">
                Card Number
              </label>
              <input
                id="card-number-input"
                type="text"
                class="shadow-sm focus:ring-[#009CDE] focus:border-[#009CDE] block w-full sm:text-sm border-gray-300 rounded-md py-3 px-4"
                placeholder="0000 0000 0000 0000"
                @input="handleCardInput"
              />
           </div>

           <div class="grid grid-cols-2 gap-4">
              <div>
                 <label class="block text-sm font-medium text-gray-700 mb-2">Expiry</label>
                 <input type="text" placeholder="MM/YY" class="shadow-sm focus:ring-[#009CDE] focus:border-[#009CDE] block w-full sm:text-sm border-gray-300 rounded-md py-3 px-4" />
              </div>
              <div>
                 <label for="card-cvv-input" class="block text-sm font-medium text-gray-700 mb-2">CVV</label>
                 <input
                   id="card-cvv-input"
                   type="text"
                   class="shadow-sm focus:ring-[#009CDE] focus:border-[#009CDE] block w-full sm:text-sm border-gray-300 rounded-md py-3 px-4"
                   placeholder="123"
                   @input="handleCVVInput"
                 />
              </div>
           </div>
           
           <div class="pt-4">
              <button
                id="submit-payment"
                @click="handleSubmit"
                class="w-full bg-[#2E7D32] text-white py-4 px-4 rounded-lg font-bold hover:bg-green-700 shadow-md transition-all flex justify-center items-center"
              >
                <svg class="h-5 w-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"></path></svg>
                Pay Now
              </button>
           </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'BILL_PAYMENT',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const handleCardInput = (e) => {
      // ACT_PAYMENT_TYPE_CARD_NUMBER
      store.card_number_entered = e.target.value
    }

    const handleCVVInput = (e) => {
      // ACT_PAYMENT_TYPE_CVV
      store.card_cvv_entered = e.target.value
    }

    const handleSubmit = async () => {
      // ACT_PAYMENT_SUBMIT
      if (store.card_number_entered.length > 0 && store.card_cvv_entered.length > 0 && store.billing_amount_due > 0) {
        store.setCurrentPageId('BILL_PAYMENT_SUCCESS')
        await router.push({ name: 'BILL_PAYMENT_SUCCESS' })
      } else {
        alert('Please complete payment details.')
      }
    }

    const handleBack = async () => {
      // ACT_PAYMENT_BACK_BILL_DETAIL
      store.setCurrentPageId('BILL_DETAIL')
      await router.push({ name: 'BILL_DETAIL' })
    }

    return {
      store,
      handleCardInput,
      handleCVVInput,
      handleSubmit,
      handleBack
    }
  }
}
</script>