<template>
  <div class="min-h-screen bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
    <div class="max-w-xl mx-auto">
      <h2 class="text-3xl font-extrabold text-gray-900 text-center mb-8">Secure Checkout</h2>
      
      <div class="bg-white shadow overflow-hidden sm:rounded-lg p-6">
        <h3 class="text-lg font-medium text-gray-900 mb-6">Payment Details</h3>
        
        <div class="space-y-6">
          <!-- Card Number -->
          <div>
            <label for="card-number-input" class="block text-sm font-medium text-gray-700">Card Number</label>
            <div class="mt-1">
              <input 
                id="card-number-input" 
                type="text" 
                class="shadow-sm focus:ring-blue-500 focus:border-blue-500 block w-full sm:text-sm border-gray-300 rounded-md p-3 border"
                placeholder="0000 0000 0000 0000"
                @input="handleCardNumber"
              >
            </div>
          </div>

          <!-- Card Name -->
          <div>
            <label for="card-name-input" class="block text-sm font-medium text-gray-700">Name on Card</label>
            <div class="mt-1">
              <input 
                id="card-name-input" 
                type="text" 
                class="shadow-sm focus:ring-blue-500 focus:border-blue-500 block w-full sm:text-sm border-gray-300 rounded-md p-3 border"
                placeholder="John Doe"
                @input="handleCardName"
              >
            </div>
          </div>

          <!-- CVV -->
          <div class="w-1/3">
            <label for="card-cvv-input" class="block text-sm font-medium text-gray-700">CVV</label>
            <div class="mt-1">
              <input 
                id="card-cvv-input" 
                type="text" 
                class="shadow-sm focus:ring-blue-500 focus:border-blue-500 block w-full sm:text-sm border-gray-300 rounded-md p-3 border"
                placeholder="123"
                @input="handleCardCvv"
              >
            </div>
          </div>

          <!-- Billing Country Dropdown -->
          <div>
            <label for="billing-country-dropdown" class="block text-sm font-medium text-gray-700">Billing Country</label>
            <div class="relative mt-1">
              <button 
                id="billing-country-dropdown"
                type="button" 
                class="relative w-full bg-white border border-gray-300 rounded-md shadow-sm pl-3 pr-10 py-3 text-left cursor-default focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                @click="toggleCountryDropdown"
              >
                <span class="block truncate">{{ selectedCountryLabel || 'Select Country' }}</span>
                <span class="absolute inset-y-0 right-0 flex items-center pr-2 pointer-events-none">
                  <svg class="h-5 w-5 text-gray-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fill-rule="evenodd" d="M10 3a1 1 0 01.707.293l3 3a1 1 0 01-1.414 1.414L10 5.414 7.707 7.707a1 1 0 01-1.414-1.414l3-3A1 1 0 0110 3zm-3.707 9.293a1 1 0 011.414 0L10 14.586l2.293-2.293a1 1 0 011.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clip-rule="evenodd" />
                  </svg>
                </span>
              </button>

              <div v-if="isCountryOpen" class="absolute z-10 mt-1 w-full bg-white shadow-lg max-h-60 rounded-md py-1 text-base ring-1 ring-black ring-opacity-5 overflow-auto focus:outline-none sm:text-sm">
                <div 
                  id="billing-country-us"
                  @click="selectCountry('us', 'United States')"
                  class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-blue-50 text-gray-900"
                >
                  United States
                </div>
                <div 
                  id="billing-country-uk"
                  @click="selectCountry('uk', 'United Kingdom')"
                  class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-blue-50 text-gray-900"
                >
                  United Kingdom
                </div>
                <div 
                  id="billing-country-in"
                  @click="selectCountry('in', 'India')"
                  class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-blue-50 text-gray-900"
                >
                  India
                </div>
              </div>
            </div>
          </div>

          <!-- Actions -->
          <div class="flex justify-between items-center mt-8 pt-6 border-t border-gray-200">
            <div 
              id="payment-back-button"
              @click="goBack"
              class="text-sm font-medium text-blue-600 hover:text-blue-500 cursor-pointer"
            >
              Back
            </div>

            <button 
              id="payment-continue-button"
              type="button" 
              class="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-blue-700 hover:bg-blue-800 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
              :disabled="!canSubmit"
              @click="goToReview"
            >
              Review Order
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'PAYMENT_DETAILS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    
    const isCountryOpen = ref(false)
    const selectedCountryLabel = ref('')

    const canSubmit = computed(() => {
      return store.card_number_filled && 
             store.card_name_filled && 
             store.card_cvv_filled && 
             store.billing_country_selected
    })

    function handleCardNumber(e) {
      store.card_number_filled = e.target.value.length > 0
    }

    function handleCardName(e) {
      store.card_name_filled = e.target.value.length > 0
    }

    function handleCardCvv(e) {
      store.card_cvv_filled = e.target.value.length > 0
    }

    function toggleCountryDropdown() {
      isCountryOpen.value = !isCountryOpen.value
    }

    function selectCountry(value, label) {
      store.billing_country_selected = true
      selectedCountryLabel.value = label
      isCountryOpen.value = false
    }

    async function goToReview() {
      if (canSubmit.value) {
        store.setCurrentPageId('ORDER_REVIEW')
        await router.push({ name: 'ORDER_REVIEW' })
      }
    }

    async function goBack() {
      store.setCurrentPageId('ENROLLMENT_OPTIONS')
      await router.push({ name: 'ENROLLMENT_OPTIONS', params: { id: store.selected_course_id } })
    }

    return {
      store,
      canSubmit,
      isCountryOpen,
      selectedCountryLabel,
      handleCardNumber,
      handleCardName,
      handleCardCvv,
      toggleCountryDropdown,
      selectCountry,
      goToReview,
      goBack
    }
  }
}
</script>