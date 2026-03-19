<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <!-- Header -->
    <header class="bg-white shadow-sm z-10">
       <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center">
         <button id="back-billing-overview" @click="handleBack" class="mr-4 text-gray-600 hover:text-gray-900">
           <svg class="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>
         </button>
         <h1 class="text-xl font-bold text-gray-900">Bill Details</h1>
       </div>
    </header>

    <main class="flex-1 max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
      <div class="bg-white rounded-lg shadow-lg overflow-hidden">
        <div class="p-6 border-b border-gray-200">
           <div class="flex justify-between items-start">
             <div>
               <h2 class="text-2xl font-bold text-gray-900">{{ bill?.description }}</h2>
               <p class="text-gray-500">Invoice #{{ bill?.id.toUpperCase() }}</p>
             </div>
             <p class="text-3xl font-bold text-gray-900">${{ bill?.amount.toFixed(2) }}</p>
           </div>
           <p class="mt-2 text-sm text-gray-500">Date of Service: {{ bill?.date }}</p>
        </div>

        <div class="p-6 space-y-8">
           <div v-if="bill?.status === 'Due'">
             <label class="block text-sm font-medium text-gray-700 mb-2">Select Payment Method</label>
             <div class="relative">
                 <button 
                   id="payment-method-dropdown" 
                   @click="toggleDropdown"
                   class="w-full bg-white border border-gray-300 rounded-md py-3 px-4 flex justify-between items-center text-left cursor-pointer focus:outline-none focus:ring-2 focus:ring-[#009CDE]"
                 >
                   <span class="block truncate">{{ selectedMethodLabel || 'Select a method' }}</span>
                   <svg class="h-5 w-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
                 </button>

                 <div v-if="dropdownOpen" class="absolute z-10 mt-1 w-full bg-white shadow-lg rounded-md py-1 text-base ring-1 ring-black ring-opacity-5 focus:outline-none sm:text-sm">
                    <div id="payment-method-card" @click="handleSelectMethod('card', 'Credit Card')" class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-gray-100">Credit Card</div>
                    <div id="payment-method-hsa" @click="handleSelectMethod('hsa', 'HSA/FSA')" class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-gray-100">HSA/FSA</div>
                    <div id="payment-method-paypal" @click="handleSelectMethod('paypal', 'PayPal')" class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-gray-100">PayPal</div>
                 </div>
              </div>
           </div>
           
           <div v-else class="bg-green-50 p-4 rounded-md">
             <p class="text-green-800 font-medium">This bill has been paid in full.</p>
           </div>
        </div>

        <div v-if="bill?.status === 'Due'" class="p-6 bg-gray-50 border-t border-gray-200">
           <button
             id="continue-to-payment"
             @click="handleContinue"
             class="w-full bg-[#005DAA] text-white py-3 px-4 rounded-lg font-bold hover:bg-[#004a87] shadow-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
             :disabled="!store.payment_method_selected"
           >
             Continue to Payment
           </button>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'BILL_DETAIL',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const dropdownOpen = ref(false)
    const selectedMethodLabel = ref('')

    const bill = computed(() => {
      return dataStore.bills.find(b => b.id === store.selected_bill_id)
    })

    const toggleDropdown = () => dropdownOpen.value = !dropdownOpen.value

    const handleSelectMethod = (value, label) => {
      // ACT_BILL_DETAIL_SELECT_PAYMENT_METHOD
      store.payment_method_selected = value
      selectedMethodLabel.value = label
      dropdownOpen.value = false
    }

    const handleContinue = async () => {
      // ACT_BILL_DETAIL_CONTINUE_PAYMENT
      // Effect: billing_amount_due = 50 (Simulated, ideally from bill amount)
      if (store.payment_method_selected) {
        store.billing_amount_due = bill.value ? bill.value.amount : 50
        store.setCurrentPageId('BILL_PAYMENT')
        await router.push({ name: 'BILL_PAYMENT' })
      }
    }

    const handleBack = async () => {
      // ACT_BILL_DETAIL_BACK_OVERVIEW
      store.setCurrentPageId('BILLING_OVERVIEW')
      await router.push({ name: 'BILLING_OVERVIEW' })
    }

    return {
      store,
      bill,
      dropdownOpen,
      selectedMethodLabel,
      toggleDropdown,
      handleSelectMethod,
      handleContinue,
      handleBack
    }
  }
}
</script>