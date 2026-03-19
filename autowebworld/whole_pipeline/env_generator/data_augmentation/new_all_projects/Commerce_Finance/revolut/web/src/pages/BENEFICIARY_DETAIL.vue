<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <!-- Top Nav -->
    <div class="bg-white px-4 py-4 shadow-sm sticky top-0 z-20 flex items-center justify-between">
      <button 
        id="back-beneficiaries" 
        @click="goBack"
        class="p-2 -ml-2 rounded-full hover:bg-gray-100 text-gray-600"
      >
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
      </button>
      <h1 class="text-lg font-bold text-gray-900">Contact</h1>
      <div class="w-10"></div>
    </div>

    <!-- Info -->
    <div v-if="beneficiary" class="flex-1 p-6 flex flex-col items-center">
      <div class="w-24 h-24 rounded-full overflow-hidden shadow-lg mb-4 border-4 border-white">
        <img :src="beneficiary.image" class="w-full h-full object-cover" alt="Profile" />
      </div>
      
      <h2 class="text-2xl font-bold text-gray-900 mb-1">{{ beneficiary.name }}</h2>
      <div class="bg-gray-200 text-gray-600 text-xs px-2 py-1 rounded-full mb-6 font-mono">
        {{ beneficiary.accountNumber }}
      </div>

      <div class="w-full bg-white rounded-2xl shadow-sm p-4 mb-8">
        <div class="flex justify-between items-center py-2 border-b border-gray-100 last:border-0">
          <span class="text-gray-500">Bank</span>
          <span class="font-medium text-gray-900">{{ beneficiary.bank }}</span>
        </div>
        <div class="flex justify-between items-center py-2">
          <span class="text-gray-500">Status</span>
          <span class="font-medium text-green-600 flex items-center gap-1">
            <span class="w-2 h-2 bg-green-500 rounded-full"></span> Verified
          </span>
        </div>
      </div>

      <button 
        id="cta-send-money"
        @click="goToTransferForm"
        class="w-full max-w-sm bg-blue-600 hover:bg-blue-700 text-white font-bold py-4 rounded-xl shadow-lg shadow-blue-200 transition-all active:scale-95 flex items-center justify-center gap-2"
      >
        <span>Send Money</span>
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path></svg>
      </button>
    </div>

    <div v-else class="flex-1 flex items-center justify-center text-gray-500">
      Beneficiary not found.
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'BENEFICIARY_DETAIL',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const beneficiary = computed(() => {
      return dataStore.beneficiaries.find(b => b.id === signatureStore.payments_selected_beneficiary_id)
    })

    const goBack = () => {
      signatureStore.setCurrentPageId('PAYMENTS_LIST')
      router.push({ name: 'PAYMENTS_LIST' })
    }

    const goToTransferForm = () => {
      // Logic from FSM: Precondition selected_beneficiary_id > 0 (satisfied by opening this page)
      // Navigate to TRANSFER_FORM
      signatureStore.setCurrentPageId('TRANSFER_FORM')
      router.push({ name: 'TRANSFER_FORM' })
    }

    return {
      beneficiary,
      goBack,
      goToTransferForm
    }
  }
}
</script>