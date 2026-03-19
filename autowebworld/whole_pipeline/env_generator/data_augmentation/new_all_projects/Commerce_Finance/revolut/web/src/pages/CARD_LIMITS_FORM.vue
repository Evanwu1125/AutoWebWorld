<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <!-- Top Nav -->
    <div class="bg-white px-4 py-4 shadow-sm sticky top-0 z-20 flex items-center justify-between">
      <button 
        id="back-limits" 
        @click="goBack"
        class="p-2 -ml-2 rounded-full hover:bg-gray-100 text-gray-600"
      >
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
      </button>
      <h1 class="text-lg font-bold text-gray-900">Spending Limits</h1>
      <div class="w-10"></div>
    </div>

    <div class="flex-1 p-6 max-w-lg mx-auto w-full flex flex-col gap-6">
      
      <!-- ATM Limit -->
      <div class="bg-white rounded-2xl shadow-sm p-6">
        <div class="flex items-center gap-3 mb-4">
           <div class="p-2 bg-blue-100 rounded-lg text-blue-600">
             <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 9V7a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2m2 4h10a2 2 0 002-2v-6a2 2 0 00-2-2H9a2 2 0 00-2 2v6a2 2 0 002 2zm7-5a2 2 0 11-4 0 2 2 0 014 0z"></path></svg>
           </div>
           <div>
             <div class="font-bold text-gray-900">ATM Withdrawals</div>
             <div class="text-xs text-gray-500">Monthly limit</div>
           </div>
        </div>
        
        <div class="mb-2 flex justify-between">
          <span class="text-sm font-medium text-gray-700">Limit: ${{ atmLimit }}</span>
          <span class="text-xs text-gray-400">Max: $20000</span>
        </div>

        <input
          id="atm-limit-slider"
          type="range"
          min="0"
          max="20000"
          step="100"
          v-model="atmLimit"
          @input="updateAtmLimit"
          class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
        />
      </div>

      <!-- POS Limit -->
      <div class="bg-white rounded-2xl shadow-sm p-6">
        <div class="flex items-center gap-3 mb-4">
           <div class="p-2 bg-purple-100 rounded-lg text-purple-600">
             <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z"></path></svg>
           </div>
           <div>
             <div class="font-bold text-gray-900">Online & In-store</div>
             <div class="text-xs text-gray-500">Monthly limit</div>
           </div>
        </div>
        
        <div class="mb-2 flex justify-between">
          <span class="text-sm font-medium text-gray-700">Limit: ${{ posLimit }}</span>
          <span class="text-xs text-gray-400">Max: $20000</span>
        </div>
        
        <input 
          id="pos-limit-slider"
          type="range" 
          min="0" 
          max="20000" 
          step="500"
          v-model="posLimit"
          @input="updatePosLimit"
          class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-purple-600"
        />
      </div>

      <div class="mt-auto">
        <button 
          id="cta-save-limits"
          @click="saveLimits"
          :disabled="!canSave"
          :class="['w-full py-4 font-bold rounded-xl shadow-lg transition-all', canSave ? 'bg-blue-600 hover:bg-blue-700 text-white shadow-blue-200 active:scale-95' : 'bg-gray-300 text-gray-500 cursor-not-allowed']"
        >
          Save Changes
        </button>
      </div>

    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'CARD_LIMITS_FORM',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const atmLimit = ref(0)
    const posLimit = ref(0)

    const canSave = computed(() => {
      // FSM precondition: length_gt 0 (meaning value > 0 for numbers treated as string or direct value)
      return atmLimit.value > 0 && posLimit.value > 0
    })

    const updateAtmLimit = (e) => {
      atmLimit.value = e.target.value
      // Store as string per FSM "placeholder" effect implication usually, 
      // but simpler to store actual value for now. 
      // FSM says effect is set "placeholder". We'll just track it in store so precondition passes.
      signatureStore.atm_limit = e.target.value
    }

    const updatePosLimit = (e) => {
      posLimit.value = e.target.value
      signatureStore.pos_limit = e.target.value
    }

    const goBack = () => {
      signatureStore.setCurrentPageId('CARD_DETAIL')
      router.push({ name: 'CARD_DETAIL' })
    }

    const saveLimits = () => {
      if (!canSave.value) return
      signatureStore.setCurrentPageId('CARD_LIMITS_SUCCESS')
      router.push({ name: 'CARD_LIMITS_SUCCESS' })
    }

    return {
      atmLimit,
      posLimit,
      canSave,
      updateAtmLimit,
      updatePosLimit,
      goBack,
      saveLimits
    }
  }
}
</script>