<template>
  <div class="min-h-screen bg-gray-50 py-12">
    <div class="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="bg-white rounded-lg shadow-lg overflow-hidden">
        <div class="px-8 py-6 border-b border-gray-200">
          <h1 class="text-2xl font-bold text-gray-900">Billing Settings</h1>
          <p class="mt-1 text-sm text-gray-500">Update your payment information.</p>
        </div>

        <div class="p-8 space-y-6">
          <!-- Card Number -->
          <div>
            <label for="input-card-number" class="block text-sm font-medium text-gray-700">Card Number</label>
            <input 
              id="input-card-number"
              type="text" 
              v-model="cardNumber"
              @input="updateCardNumber"
              class="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm p-2 border"
              placeholder="XXXX XXXX XXXX XXXX"
            >
          </div>

          <!-- Expiry -->
          <div>
            <label for="input-card-expiry" class="block text-sm font-medium text-gray-700">Expiry Date</label>
            <input 
              id="input-card-expiry"
              type="text" 
              v-model="expiry"
              @input="updateExpiry"
              class="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm p-2 border"
              placeholder="MM/YY"
            >
          </div>

          <!-- Country Dropdown -->
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Billing Country</label>
            <div class="relative" id="billing-country-dropdown">
              <button 
                type="button"
                @click="toggleDropdown"
                class="bg-white relative w-full border border-gray-300 rounded-md shadow-sm pl-3 pr-10 py-2 text-left cursor-default focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
              >
                <span class="block truncate">{{ countryLabel }}</span>
                <span class="absolute inset-y-0 right-0 flex items-center pr-2 pointer-events-none">
                  <svg class="h-5 w-5 text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                    <path fill-rule="evenodd" d="M10 3a1 1 0 01.707.293l3 3a1 1 0 01-1.414 1.414L10 5.414 7.707 7.707a1 1 0 01-1.414-1.414l3-3A1 1 0 0110 3zm-3.707 9.293a1 1 0 011.414 0L10 14.586l2.293-2.293a1 1 0 011.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clip-rule="evenodd" />
                  </svg>
                </span>
              </button>

              <div v-if="dropdownOpen" class="absolute z-10 mt-1 w-full bg-white shadow-lg max-h-60 rounded-md py-1 text-base ring-1 ring-black ring-opacity-5 overflow-auto focus:outline-none sm:text-sm">
                <div 
                  id="billing-country-us" 
                  @click="selectCountry('us', 'United States')"
                  class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-blue-50 text-gray-900"
                >
                  <span class="font-normal block truncate">United States</span>
                </div>
                <div 
                  id="billing-country-uk" 
                  @click="selectCountry('uk', 'United Kingdom')"
                  class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-blue-50 text-gray-900"
                >
                  <span class="font-normal block truncate">United Kingdom</span>
                </div>
                <div 
                  id="billing-country-de" 
                  @click="selectCountry('de', 'Germany')"
                  class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-blue-50 text-gray-900"
                >
                  <span class="font-normal block truncate">Germany</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Footer -->
        <div class="bg-gray-50 px-8 py-6 flex justify-between items-center">
          <button 
            id="btn-billing-back"
            @click="goBack"
            class="text-sm text-gray-600 hover:text-gray-900 font-medium"
          >
            Cancel
          </button>
          <button 
            id="btn-save-billing"
            @click="save"
            :disabled="!isValid"
            class="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Update Billing
          </button>
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
  name: 'BILLING_SETTINGS',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const cardNumber = ref('')
    const expiry = ref('')
    const dropdownOpen = ref(false)
    const countryLabel = ref('Select Country')

    function updateCardNumber() {
      signatureStore.card_number_set = true
    }

    function updateExpiry() {
      signatureStore.card_expiry_set = true
    }

    function toggleDropdown() {
      dropdownOpen.value = !dropdownOpen.value
    }

    function selectCountry(val, label) {
      signatureStore.billing_country_selected = true
      countryLabel.value = label
      dropdownOpen.value = false
    }

    const isValid = computed(() => {
      return cardNumber.value.length > 0 && 
             expiry.value.length > 0 && 
             signatureStore.billing_country_selected
    })

    function save() {
      if (isValid.value) {
        signatureStore.setCurrentPageId('ACCOUNT_BILLING_UPDATED_SUCCESS')
        router.push({ name: 'ACCOUNT_BILLING_UPDATED_SUCCESS' })
      }
    }

    function goBack() {
      signatureStore.setCurrentPageId('ACCOUNT_SETTINGS')
      router.push({ name: 'ACCOUNT_SETTINGS' })
    }

    return {
      cardNumber,
      expiry,
      dropdownOpen,
      countryLabel,
      updateCardNumber,
      updateExpiry,
      toggleDropdown,
      selectCountry,
      isValid,
      save,
      goBack
    }
  }
}
</script>