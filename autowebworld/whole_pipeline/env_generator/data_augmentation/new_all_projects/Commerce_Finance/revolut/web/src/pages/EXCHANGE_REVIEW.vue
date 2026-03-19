<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <!-- Top Nav -->
    <div class="bg-white px-4 py-4 shadow-sm sticky top-0 z-20 flex items-center justify-between">
      <button 
        id="back-exchange-review" 
        @click="goBack"
        class="p-2 -ml-2 rounded-full hover:bg-gray-100 text-gray-600"
      >
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
      </button>
      <h1 class="text-lg font-bold text-gray-900">Confirm Order</h1>
      <div class="w-10"></div>
    </div>

    <div v-if="pair" class="flex-1 p-6 max-w-lg mx-auto w-full flex flex-col">
      
      <div class="bg-white rounded-2xl shadow-sm p-6 mb-8 flex flex-col items-center">
        <div class="text-gray-500 mb-2 font-medium">You sell</div>
        <div class="text-4xl font-extrabold text-gray-900 mb-2">{{ formatCurrency(sellAmount, pair.from) }}</div>
        
        <div class="my-4 text-gray-300">
           <svg class="w-6 h-6 animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 14l-7 7m0 0l-7-7m7 7V3"></path></svg>
        </div>

        <div class="text-gray-500 mb-2 font-medium">You buy</div>
        <div class="text-4xl font-extrabold text-green-600 mb-8">{{ formatCurrency(buyAmount, pair.to) }}</div>
        
        <div class="w-full bg-gray-50 rounded-xl p-4 space-y-3">
          <div class="flex justify-between items-center text-sm">
            <span class="text-gray-500">Rate</span>
            <span class="font-mono font-bold text-gray-900">1 {{ pair.from }} = {{ pair.rate }} {{ pair.to }}</span>
          </div>
          <div class="flex justify-between items-center text-sm">
            <span class="text-gray-500">Fee</span>
            <span class="font-bold text-gray-900">No fee</span>
          </div>
        </div>
      </div>

      <div class="mt-auto">
        <p class="text-xs text-center text-gray-400 mb-4">
          By clicking confirm, you agree to the Terms & Conditions. Rates refresh every 30 seconds.
        </p>
        <button 
          id="cta-confirm-exchange"
          @click="submitExchange"
          class="w-full py-4 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-xl shadow-lg shadow-blue-200 transition-all active:scale-95"
        >
          Confirm Exchange
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
  name: 'EXCHANGE_REVIEW',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const pair = computed(() => {
      return dataStore.exchangePairs.find(p => p.id === signatureStore.exchange_selected_pair_id)
    })
    
    const sellAmount = computed(() => signatureStore.sell_amount)
    const buyAmount = computed(() => signatureStore.buy_amount)

    const formatCurrency = (val, curr) => {
      if (!val) return '0.00'
      // Simple format
      return `${parseFloat(val).toFixed(2)} ${curr}`
    }

    const goBack = () => {
      signatureStore.setCurrentPageId('EXCHANGE_FORM')
      router.push({ name: 'EXCHANGE_FORM' })
    }

    const submitExchange = () => {
      signatureStore.setCurrentPageId('EXCHANGE_SUCCESS')
      router.push({ name: 'EXCHANGE_SUCCESS' })
    }

    return {
      pair,
      sellAmount,
      buyAmount,
      formatCurrency,
      goBack,
      submitExchange
    }
  }
}
</script>