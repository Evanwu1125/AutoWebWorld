<template>
  <div class="min-h-screen bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
    <div class="max-w-3xl mx-auto">
      <h2 class="text-3xl font-extrabold text-gray-900 text-center mb-8">Choose Enrollment Option</h2>
      
      <div class="bg-white shadow overflow-hidden sm:rounded-lg mb-6">
        <div class="px-4 py-5 sm:px-6">
          <h3 class="text-lg leading-6 font-medium text-gray-900">Purchase Course</h3>
          <p class="mt-1 max-w-2xl text-sm text-gray-500">Select a pricing plan to continue.</p>
        </div>
        
        <div class="border-t border-gray-200 px-4 py-5 sm:p-6">
          <!-- Dropdown for Pricing -->
          <div class="relative w-full mb-6">
            <button 
              id="pricing-dropdown"
              type="button" 
              class="relative w-full bg-white border border-gray-300 rounded-md shadow-sm pl-3 pr-10 py-3 text-left cursor-default focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
              @click="toggleDropdown"
            >
              <span class="block truncate">
                {{ selectedOptionLabel || 'Select an option...' }}
              </span>
              <span class="absolute inset-y-0 right-0 flex items-center pr-2 pointer-events-none">
                <svg class="h-5 w-5 text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                  <path fill-rule="evenodd" d="M10 3a1 1 0 01.707.293l3 3a1 1 0 01-1.414 1.414L10 5.414 7.707 7.707a1 1 0 01-1.414-1.414l3-3A1 1 0 0110 3zm-3.707 9.293a1 1 0 011.414 0L10 14.586l2.293-2.293a1 1 0 011.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clip-rule="evenodd" />
                </svg>
              </span>
            </button>

            <div v-if="isOpen" class="absolute z-10 mt-1 w-full bg-white shadow-lg max-h-60 rounded-md py-1 text-base ring-1 ring-black ring-opacity-5 overflow-auto focus:outline-none sm:text-sm">
              <div 
                id="pricing-option-one-time"
                @click="selectOption('one-time', 'One-time Purchase ($49.99)')"
                class="cursor-pointer select-none relative py-3 pl-3 pr-9 hover:bg-blue-50 text-gray-900"
              >
                <span class="font-normal block truncate">One-time Purchase ($49.99)</span>
                <span class="text-xs text-gray-500 block">Full lifetime access</span>
              </div>
              <div 
                id="pricing-option-subscription"
                @click="selectOption('subscription', 'Monthly Subscription ($39.00/mo)')"
                class="cursor-pointer select-none relative py-3 pl-3 pr-9 hover:bg-blue-50 text-gray-900"
              >
                <span class="font-normal block truncate">Monthly Subscription ($39.00/mo)</span>
                <span class="text-xs text-gray-500 block">Cancel anytime</span>
              </div>
            </div>
          </div>

          <div class="mt-8 flex justify-between items-center">
            <div 
              id="enrollment-options-back"
              @click="goBack"
              class="text-sm font-medium text-blue-600 hover:text-blue-500 cursor-pointer"
            >
              Back
            </div>

            <button 
              id="continue-to-payment-button"
              type="button" 
              class="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-blue-700 hover:bg-blue-800 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
              :disabled="!store.selected_pricing_option"
              @click="goToPayment"
            >
              Continue to Payment
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'ENROLLMENT_OPTIONS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    
    const isOpen = ref(false)
    const selectedOptionLabel = ref('')

    function toggleDropdown() {
      isOpen.value = !isOpen.value
    }

    function selectOption(value, label) {
      store.selected_pricing_option = value
      selectedOptionLabel.value = label
      isOpen.value = false
    }

    async function goToPayment() {
      if (store.selected_pricing_option && store.selected_course_id) {
        store.setCurrentPageId('PAYMENT_DETAILS')
        await router.push({ name: 'PAYMENT_DETAILS' })
      }
    }

    async function goBack() {
      store.setCurrentPageId('COURSE_DETAIL')
      await router.push({ name: 'COURSE_DETAIL', params: { id: store.selected_course_id } })
    }

    return {
      store,
      isOpen,
      selectedOptionLabel,
      toggleDropdown,
      selectOption,
      goToPayment,
      goBack
    }
  }
}
</script>