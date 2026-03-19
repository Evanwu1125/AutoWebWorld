<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <!-- Top Nav -->
    <div class="bg-white px-4 py-4 shadow-sm sticky top-0 z-20 flex items-center justify-between">
      <button 
        id="back-topup-review" 
        @click="goBack"
        class="p-2 -ml-2 rounded-full hover:bg-gray-100 text-gray-600"
      >
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
      </button>
      <h1 class="text-lg font-bold text-gray-900">Review</h1>
      <div class="w-10"></div>
    </div>

    <div class="flex-1 p-6 max-w-lg mx-auto w-full flex flex-col">
      
      <div class="bg-white rounded-2xl shadow-sm p-6 mb-8 flex flex-col items-center">
        <div class="text-gray-500 mb-2 font-medium">Adding</div>
        <div class="text-4xl font-extrabold text-gray-900 mb-8">${{ amount }}</div>
        
        <div class="w-full space-y-4">
          <div class="flex justify-between items-center py-2 border-b border-gray-100">
            <span class="text-gray-500">From</span>
            <div class="flex items-center gap-2">
              <img v-if="method" :src="method.image" class="w-6 h-6 object-contain" />
              <span class="font-bold text-gray-900">{{ method ? method.name : 'Unknown' }}</span>
            </div>
          </div>
          <div class="flex justify-between items-center py-2 border-b border-gray-100">
            <span class="text-gray-500">Fee</span>
            <span class="font-bold text-green-600">Free</span>
          </div>
           <div class="flex justify-between items-center py-2 border-b border-gray-100">
            <span class="text-gray-500">Total charge</span>
            <span class="font-bold text-gray-900">${{ amount }}</span>
          </div>
        </div>
      </div>

      <div class="mt-auto">
        <button 
          id="cta-confirm-topup"
          @click="submitTopup"
          class="w-full py-4 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-xl shadow-lg shadow-blue-200 transition-all active:scale-95 flex items-center justify-center gap-2"
        >
          <span>Add ${{ amount }} securely</span>
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"></path></svg>
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
  name: 'TOPUP_REVIEW',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const amount = computed(() => signatureStore.topup_amount)
    
    const method = computed(() => {
      return dataStore.topupMethods.find(m => m.id === signatureStore.topup_selected_method_id)
    })

    const goBack = () => {
      signatureStore.setCurrentPageId('TOPUP_FORM')
      router.push({ name: 'TOPUP_FORM' })
    }

    const submitTopup = () => {
      signatureStore.setCurrentPageId('TOPUP_SUCCESS')
      router.push({ name: 'TOPUP_SUCCESS' })
    }

    return {
      amount,
      method,
      goBack,
      submitTopup
    }
  }
}
</script>