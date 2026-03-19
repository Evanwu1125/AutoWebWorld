<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <!-- Top Nav -->
    <div class="bg-white px-4 py-4 shadow-sm sticky top-0 z-20 flex items-center justify-between">
      <button 
        id="back-exchange-form" 
        @click="goBack"
        class="p-2 -ml-2 rounded-full hover:bg-gray-100 text-gray-600"
      >
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
      </button>
      <h1 class="text-lg font-bold text-gray-900">Exchange</h1>
      <div class="w-10"></div>
    </div>

    <div v-if="pair" class="flex-1 p-6 max-w-lg mx-auto w-full flex flex-col">
      
      <!-- Rate Info -->
      <div class="text-center mb-8">
        <div class="text-gray-500 font-medium mb-1">Market Rate</div>
        <div class="text-3xl font-extrabold text-gray-900">1 {{ pair.from }} = {{ pair.rate }} {{ pair.to }}</div>
        <div class="text-xs text-green-600 font-bold mt-1">Live updates</div>
      </div>

      <!-- Inputs Container -->
      <div class="bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden mb-8 relative">
        
        <!-- Sell Section -->
        <div class="p-6 border-b border-gray-100">
          <div class="flex justify-between items-center mb-2">
            <span class="text-gray-500 font-medium">Sell</span>
            <span class="text-gray-900 font-bold flex items-center gap-2">
               <img :src="`https://flagcdn.com/w20/${pair.from.substring(0,2).toLowerCase()}.png`" class="w-5 h-3 shadow-sm" alt="flag" />
               {{ pair.from }}
            </span>
          </div>
          <div class="relative">
            <input 
              id="input-sell-amount"
              type="number" 
              v-model="sellAmount"
              @input="updateSell"
              placeholder="0"
              class="w-full text-4xl font-bold text-gray-900 placeholder-gray-300 outline-none bg-transparent"
            />
          </div>
        </div>

        <!-- Swap Icon -->
        <div class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-white rounded-full p-2 shadow-md border border-gray-100">
           <svg class="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 14l-7 7m0 0l-7-7m7 7V3"></path></svg>
        </div>

        <!-- Buy Section -->
        <div class="p-6 bg-gray-50">
           <div class="flex justify-between items-center mb-2">
            <span class="text-gray-500 font-medium">Buy</span>
            <span class="text-gray-900 font-bold flex items-center gap-2">
               <img :src="`https://flagcdn.com/w20/${pair.to.substring(0,2).toLowerCase()}.png`" class="w-5 h-3 shadow-sm" alt="flag" />
               {{ pair.to }}
            </span>
          </div>
          <div class="relative">
             <input 
              id="input-buy-amount"
              type="number" 
              v-model="buyAmount"
              @input="updateBuy"
              placeholder="0"
              class="w-full text-4xl font-bold text-green-600 placeholder-gray-300 outline-none bg-transparent"
            />
          </div>
        </div>
      </div>

      <!-- Continue Button -->
      <div class="mt-auto">
        <button 
          id="cta-continue-exchange"
          @click="continueToReview"
          :disabled="!isValid"
          :class="['w-full py-4 rounded-xl font-bold shadow-lg transition-all', isValid ? 'bg-blue-600 hover:bg-blue-700 text-white shadow-blue-200 active:scale-95' : 'bg-gray-300 text-gray-500 cursor-not-allowed']"
        >
          Review Order
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
  name: 'EXCHANGE_FORM',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const pair = computed(() => {
      return dataStore.exchangePairs.find(p => p.id === signatureStore.exchange_selected_pair_id)
    })

    const sellAmount = ref('')
    const buyAmount = ref('')

    const isValid = computed(() => {
      return sellAmount.value && parseFloat(sellAmount.value) > 0
    })

    // Simple reactive calculation
    const updateSell = (e) => {
      const val = e.target.value
      sellAmount.value = val
      signatureStore.sell_amount = val // Sync store
      
      if (val && pair.value) {
        buyAmount.value = (parseFloat(val) * pair.value.rate).toFixed(2)
        signatureStore.buy_amount = buyAmount.value
      } else {
        buyAmount.value = ''
        signatureStore.buy_amount = ''
      }
    }

    const updateBuy = (e) => {
      const val = e.target.value
      buyAmount.value = val
      signatureStore.buy_amount = val
      
      if (val && pair.value) {
        sellAmount.value = (parseFloat(val) / pair.value.rate).toFixed(2)
        signatureStore.sell_amount = sellAmount.value
      } else {
        sellAmount.value = ''
        signatureStore.sell_amount = ''
      }
    }

    const goBack = () => {
      signatureStore.setCurrentPageId('EXCHANGE_DASHBOARD')
      router.push({ name: 'EXCHANGE_DASHBOARD' })
    }

    const continueToReview = () => {
      if (!isValid.value) return
      signatureStore.setCurrentPageId('EXCHANGE_REVIEW')
      router.push({ name: 'EXCHANGE_REVIEW' })
    }

    return {
      pair,
      sellAmount,
      buyAmount,
      isValid,
      updateSell,
      updateBuy,
      goBack,
      continueToReview
    }
  }
}
</script>