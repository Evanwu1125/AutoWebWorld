<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <!-- Top Nav -->
    <div class="bg-white px-4 py-4 shadow-sm sticky top-0 z-20 flex items-center justify-between">
      <button 
        id="back-topup-form" 
        @click="goBack"
        class="p-2 -ml-2 rounded-full hover:bg-gray-100 text-gray-600"
      >
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
      </button>
      <h1 class="text-lg font-bold text-gray-900">Amount</h1>
      <div class="w-10"></div>
    </div>

    <div class="flex-1 p-6 max-w-lg mx-auto w-full flex flex-col">
      
      <div class="flex items-center gap-3 mb-8 bg-white p-4 rounded-xl shadow-sm border border-gray-100">
        <div v-if="method" class="w-10 h-10 rounded-lg overflow-hidden bg-gray-50 p-1 border border-gray-100">
           <img :src="method.image" class="w-full h-full object-contain" />
        </div>
        <div>
          <div class="text-xs text-gray-500">Adding money from</div>
          <div v-if="method" class="font-bold text-gray-900">{{ method.name }}</div>
        </div>
      </div>

      <div class="bg-white rounded-2xl shadow-sm p-8 mb-8 flex flex-col items-center justify-center py-12">
        <div class="relative w-full text-center">
          <span class="text-4xl font-bold text-gray-900 mr-1">$</span>
          <input 
            id="input-topup-amount"
            type="number" 
            v-model="amount"
            @input="updateAmount"
            placeholder="0"
            class="w-40 text-5xl font-extrabold text-gray-900 placeholder-gray-200 outline-none bg-transparent text-center"
            autofocus
          />
        </div>
      </div>

      <div class="mt-auto">
        <button 
          id="cta-continue-topup"
          @click="continueToReview"
          :disabled="!isValid"
          :class="['w-full py-4 rounded-xl font-bold shadow-lg transition-all', isValid ? 'bg-blue-600 hover:bg-blue-700 text-white shadow-blue-200 active:scale-95' : 'bg-gray-300 text-gray-500 cursor-not-allowed']"
        >
          Continue
        </button>
      </div>

    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'TOPUP_FORM',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const amount = ref('')

    const method = computed(() => {
      return dataStore.topupMethods.find(m => m.id === signatureStore.topup_selected_method_id)
    })

    const isValid = computed(() => {
      return amount.value && parseFloat(amount.value) > 0
    })

    const updateAmount = (e) => {
      amount.value = e.target.value
      signatureStore.topup_amount = e.target.value
    }

    const goBack = () => {
      signatureStore.setCurrentPageId('TOPUP_METHOD_LIST')
      router.push({ name: 'TOPUP_METHOD_LIST' })
    }

    const continueToReview = () => {
      if (!isValid.value) return
      signatureStore.setCurrentPageId('TOPUP_REVIEW')
      router.push({ name: 'TOPUP_REVIEW' })
    }

    return {
      amount,
      method,
      isValid,
      updateAmount,
      goBack,
      continueToReview
    }
  }
}
</script>