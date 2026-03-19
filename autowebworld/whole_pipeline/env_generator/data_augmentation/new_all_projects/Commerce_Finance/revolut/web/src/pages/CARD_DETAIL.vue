<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <!-- Top Nav -->
    <div class="bg-white px-4 py-4 shadow-sm sticky top-0 z-20 flex items-center justify-between">
      <button 
        id="back-cards" 
        @click="goBack"
        class="p-2 -ml-2 rounded-full hover:bg-gray-100 text-gray-600"
      >
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
      </button>
      <h1 class="text-lg font-bold text-gray-900">Card Settings</h1>
      <div class="w-10"></div>
    </div>

    <div v-if="card" class="flex-1 p-6 flex flex-col items-center max-w-lg mx-auto w-full">
      
      <!-- Card Visual -->
      <div class="relative w-full h-56 rounded-2xl shadow-xl overflow-hidden mb-8 transform hover:scale-105 transition-transform duration-300">
        <img :src="card.image" class="w-full h-full object-cover" />
        <div class="absolute inset-0 bg-black/10"></div>
        <div class="absolute inset-0 p-6 flex flex-col justify-between text-white">
          <div class="flex justify-between items-start">
             <div class="font-bold text-lg tracking-wide">{{ card.nickname }}</div>
             <div class="font-bold italic opacity-90">{{ card.scheme }}</div>
          </div>
          <div>
            <div class="text-xl font-mono tracking-widest mb-1">•••• •••• •••• {{ card.last4 }}</div>
            <div class="flex justify-between items-end">
              <div class="text-sm opacity-80">Exp {{ card.expiry }}</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Actions Grid -->
      <div class="grid grid-cols-2 gap-4 w-full mb-8">
        
        <!-- Freeze Action -->
        <button 
          id="cta-freeze-card"
          @click="goToFreeze"
          class="flex flex-col items-center justify-center p-6 bg-white rounded-2xl shadow-sm border border-transparent hover:border-red-200 hover:shadow-md transition-all group"
        >
          <div class="w-12 h-12 bg-red-50 text-red-500 rounded-full flex items-center justify-center mb-3 group-hover:scale-110 transition-transform">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"></path></svg>
          </div>
          <span class="font-bold text-gray-900">Freeze Card</span>
          <span class="text-xs text-gray-500 mt-1">Temporarily block</span>
        </button>

        <!-- Limits Action -->
        <button 
          id="cta-edit-limits"
          @click="goToLimits"
          class="flex flex-col items-center justify-center p-6 bg-white rounded-2xl shadow-sm border border-transparent hover:border-blue-200 hover:shadow-md transition-all group"
        >
          <div class="w-12 h-12 bg-blue-50 text-blue-500 rounded-full flex items-center justify-center mb-3 group-hover:scale-110 transition-transform">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"></path></svg>
          </div>
          <span class="font-bold text-gray-900">Limits</span>
          <span class="text-xs text-gray-500 mt-1">Control spending</span>
        </button>

      </div>

      <!-- Info Section -->
      <div class="w-full bg-white rounded-2xl p-5 shadow-sm space-y-4">
        <div class="flex justify-between items-center">
          <span class="text-gray-500">Status</span>
          <span :class="['font-bold px-2 py-1 rounded text-sm', card.status === 'Active' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700']">
            {{ card.status }}
          </span>
        </div>
        <div class="flex justify-between items-center">
          <span class="text-gray-500">Monthly Limit</span>
          <span class="font-bold text-gray-900">${{ card.limit }}</span>
        </div>
         <div class="flex justify-between items-center">
          <span class="text-gray-500">CVV</span>
          <span class="font-bold text-gray-900">•••</span>
        </div>
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
  name: 'CARD_DETAIL',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const card = computed(() => {
      return dataStore.cards.find(c => c.id === signatureStore.cards_selected_card_id)
    })

    const goBack = () => {
      signatureStore.setCurrentPageId('CARDS_LIST')
      router.push({ name: 'CARDS_LIST' })
    }

    const goToFreeze = () => {
      signatureStore.setCurrentPageId('CARD_FREEZE_FORM')
      router.push({ name: 'CARD_FREEZE_FORM' })
    }

    const goToLimits = () => {
      signatureStore.setCurrentPageId('CARD_LIMITS_FORM')
      router.push({ name: 'CARD_LIMITS_FORM' })
    }

    return {
      card,
      goBack,
      goToFreeze,
      goToLimits
    }
  }
}
</script>