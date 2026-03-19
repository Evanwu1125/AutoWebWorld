<template>
  <div class="min-h-screen bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
    <div class="max-w-xl mx-auto">
      <h2 class="text-3xl font-extrabold text-gray-900 text-center mb-8">Enrollment Payment</h2>
      
      <div class="bg-white shadow overflow-hidden sm:rounded-lg p-6">
        <h3 class="text-lg font-medium text-gray-900 mb-6">Payment Information</h3>
        
        <div class="space-y-6">
          <div>
            <label for="pro-cert-card-number-input" class="block text-sm font-medium text-gray-700">Card Number</label>
            <div class="mt-1">
              <input 
                id="pro-cert-card-number-input" 
                type="text" 
                class="shadow-sm focus:ring-blue-500 focus:border-blue-500 block w-full sm:text-sm border-gray-300 rounded-md p-3 border"
                placeholder="0000 0000 0000 0000"
                @input="handleCardNumber"
              >
            </div>
          </div>

          <div>
            <label for="pro-cert-card-name-input" class="block text-sm font-medium text-gray-700">Name on Card</label>
            <div class="mt-1">
              <input 
                id="pro-cert-card-name-input" 
                type="text" 
                class="shadow-sm focus:ring-blue-500 focus:border-blue-500 block w-full sm:text-sm border-gray-300 rounded-md p-3 border"
                placeholder="John Doe"
                @input="handleCardName"
              >
            </div>
          </div>

          <div class="w-1/3">
            <label for="pro-cert-card-cvv-input" class="block text-sm font-medium text-gray-700">CVV</label>
            <div class="mt-1">
              <input 
                id="pro-cert-card-cvv-input" 
                type="text" 
                class="shadow-sm focus:ring-blue-500 focus:border-blue-500 block w-full sm:text-sm border-gray-300 rounded-md p-3 border"
                placeholder="123"
                @input="handleCardCvv"
              >
            </div>
          </div>

          <div class="flex justify-between items-center mt-8 pt-6 border-t border-gray-200">
            <div 
              id="pro-cert-payment-back-button"
              @click="goBack"
              class="text-sm font-medium text-blue-600 hover:text-blue-500 cursor-pointer"
            >
              Back
            </div>

            <button 
              id="pro-cert-payment-submit-button"
              type="button" 
              class="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-blue-700 hover:bg-blue-800 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
              :disabled="!canSubmit"
              @click="submitPayment"
            >
              Complete Enrollment
            </button>
          </div>
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
  name: 'PROFESSIONAL_CERT_ENROLL_PAYMENT',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const canSubmit = computed(() => store.card_number_filled && store.card_name_filled && store.card_cvv_filled)

    function handleCardNumber(e) {
      store.card_number_filled = e.target.value.length > 0
    }

    function handleCardName(e) {
      store.card_name_filled = e.target.value.length > 0
    }

    function handleCardCvv(e) {
      store.card_cvv_filled = e.target.value.length > 0
    }

    async function submitPayment() {
      if (canSubmit.value) {
        store.setCurrentPageId('ENROLL_PROFESSIONAL_CERT_SUCCESS')
        await router.push({ name: 'ENROLL_PROFESSIONAL_CERT_SUCCESS' })
      }
    }

    async function goBack() {
      store.setCurrentPageId('PROFESSIONAL_CERT_DETAIL')
      await router.push({ name: 'PROFESSIONAL_CERT_DETAIL', params: { id: store.selected_pro_cert_id } })
    }

    return {
      store,
      canSubmit,
      handleCardNumber,
      handleCardName,
      handleCardCvv,
      submitPayment,
      goBack
    }
  }
}
</script>