<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <!-- Header -->
    <header class="bg-white shadow-sm z-10">
       <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center">
         <button id="back-renewal-detail" @click="handleBack" class="mr-4 text-gray-600 hover:text-gray-900">
           <svg class="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>
         </button>
         <h1 class="text-xl font-bold text-gray-900">Review Renewal</h1>
       </div>
    </header>

    <main class="flex-1 max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
      <div class="bg-white rounded-lg shadow-lg overflow-hidden">
        <div class="p-6">
           <h2 class="text-lg font-bold text-gray-900 mb-6">Confirm Request</h2>
           
           <div class="space-y-4">
              <div class="bg-gray-50 p-4 rounded-md">
                 <p class="text-sm font-medium text-gray-500">Medication</p>
                 <p class="text-lg font-bold text-[#005DAA]">{{ prescription?.name }}</p>
                 <p class="text-sm text-gray-600">{{ prescription?.dosage }}</p>
              </div>

              <div class="bg-gray-50 p-4 rounded-md">
                 <p class="text-sm font-medium text-gray-500">Notes</p>
                 <p class="text-base text-gray-900">{{ store.renewal_notes || 'No notes added.' }}</p>
              </div>
           </div>
        </div>

        <div class="p-6 bg-gray-50">
           <button
             id="confirm-renewal"
             @click="handleConfirm"
             class="w-full bg-[#2E7D32] text-white py-4 px-4 rounded-lg font-bold hover:bg-green-700 shadow-lg transition-transform transform hover:-translate-y-1"
           >
             Confirm Renewal
           </button>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'PRESCRIPTION_RENEWAL_REVIEW',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const prescription = computed(() => {
      return dataStore.prescriptions.find(r => r.id === store.selected_prescription_id)
    })

    const handleConfirm = async () => {
      // ACT_RX_RENEWAL_CONFIRM
      store.setCurrentPageId('PRESCRIPTION_RENEWAL_SUCCESS')
      await router.push({ name: 'PRESCRIPTION_RENEWAL_SUCCESS' })
    }

    const handleBack = async () => {
      // ACT_RX_RENEWAL_BACK_DETAIL
      store.setCurrentPageId('PRESCRIPTION_DETAIL')
      await router.push({ name: 'PRESCRIPTION_DETAIL' })
    }

    return {
      store,
      prescription,
      handleConfirm,
      handleBack
    }
  }
}
</script>