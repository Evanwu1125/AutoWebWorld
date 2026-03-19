<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <!-- Top Nav -->
    <div class="bg-white px-4 py-4 shadow-sm sticky top-0 z-20 flex items-center justify-between">
      <button 
        id="back-transfer-review" 
        @click="goBack"
        class="p-2 -ml-2 rounded-full hover:bg-gray-100 text-gray-600"
      >
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
      </button>
      <h1 class="text-lg font-bold text-gray-900">Review Transfer</h1>
      <div class="w-10"></div>
    </div>

    <div class="flex-1 p-6 max-w-lg mx-auto w-full flex flex-col">
      
      <div class="bg-white rounded-2xl shadow-sm p-6 mb-8 flex flex-col items-center">
        <div class="text-gray-500 mb-2 font-medium">Sending</div>
        <div class="text-4xl font-extrabold text-gray-900 mb-6">${{ amount }}</div>
        
        <div class="w-full space-y-4">
          <div class="flex justify-between items-center py-2 border-b border-gray-100">
            <span class="text-gray-500">To</span>
            <span class="font-bold text-gray-900">{{ beneficiaryName }}</span>
          </div>
          <div class="flex justify-between items-center py-2 border-b border-gray-100">
            <span class="text-gray-500">From</span>
            <span class="font-bold text-gray-900">{{ fromAccountName }}</span>
          </div>
          <div class="flex justify-between items-center py-2 border-b border-gray-100">
            <span class="text-gray-500">Reference</span>
            <span class="font-medium text-gray-900">{{ reference || 'None' }}</span>
          </div>
          <div class="flex justify-between items-center py-2">
            <span class="text-gray-500">Fee</span>
            <span class="font-bold text-green-600">Free</span>
          </div>
        </div>
      </div>

      <div class="mt-auto">
        <button 
          id="cta-confirm-transfer"
          @click="submitTransfer"
          class="w-full py-4 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-xl shadow-lg shadow-blue-200 transition-all active:scale-95 flex items-center justify-center gap-2"
        >
          <span>Send ${{ amount }}</span>
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
        </button>
      </div>

    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'TRANSFER_REVIEW.vue', // Should be TRANSFER_REVIEW based on logic, but preserving standard name
  // Actually standard says component name should match page ID.
  name: 'TRANSFER_REVIEW',
  
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const amount = computed(() => signatureStore.transfer_amount)
    const reference = computed(() => signatureStore.transfer_reference)
    
    const beneficiaryName = computed(() => {
      const b = dataStore.beneficiaries.find(i => i.id === signatureStore.payments_selected_beneficiary_id)
      return b ? b.name : 'Unknown'
    })

    const fromAccountName = computed(() => {
      const a = dataStore.accounts.find(i => i.id === signatureStore.from_account_id)
      return a ? a.name : 'Unknown'
    })

    const goBack = () => {
      signatureStore.setCurrentPageId('TRANSFER_FORM')
      router.push({ name: 'TRANSFER_FORM' })
    }

    const submitTransfer = () => {
      signatureStore.setCurrentPageId('TRANSFER_SUCCESS')
      router.push({ name: 'TRANSFER_SUCCESS' })
    }

    return {
      amount,
      reference,
      beneficiaryName,
      fromAccountName,
      goBack,
      submitTransfer
    }
  }
}
</script>