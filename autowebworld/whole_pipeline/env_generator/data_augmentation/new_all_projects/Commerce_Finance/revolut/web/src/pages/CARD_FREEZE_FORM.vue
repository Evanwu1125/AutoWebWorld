<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <!-- Top Nav -->
    <div class="bg-white px-4 py-4 shadow-sm sticky top-0 z-20 flex items-center justify-between">
      <button 
        id="back-freeze" 
        @click="goBack"
        class="p-2 -ml-2 rounded-full hover:bg-gray-100 text-gray-600"
      >
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
      </button>
      <h1 class="text-lg font-bold text-gray-900">Freeze Card</h1>
      <div class="w-10"></div>
    </div>

    <div class="flex-1 p-6 max-w-lg mx-auto w-full flex flex-col">
      
      <div class="bg-white rounded-2xl shadow-sm p-6 mb-8 text-center">
        <div class="w-16 h-16 bg-red-100 text-red-500 rounded-full flex items-center justify-center mx-auto mb-4">
           <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"></path></svg>
        </div>
        <h2 class="text-xl font-bold text-gray-900 mb-2">Are you sure?</h2>
        <p class="text-gray-500">Freezing this card will block all new transactions. You can unfreeze it anytime.</p>
      </div>

      <div class="mb-8">
        <label class="block text-sm font-medium text-gray-700 mb-2">Reason (Optional)</label>
        <input 
          id="input-freeze-reason"
          type="text" 
          v-model="reason"
          @input="updateReason"
          placeholder="e.g. Lost card"
          class="w-full px-4 py-3 bg-white border border-gray-300 rounded-xl text-gray-900 focus:ring-2 focus:ring-red-500 focus:border-red-500 outline-none transition-all shadow-sm"
        />
      </div>

      <div class="mt-auto">
        <button 
          id="cta-confirm-freeze"
          @click="confirmFreeze"
          :disabled="!reason"
          :class="['w-full py-4 font-bold rounded-xl shadow-lg transition-all', reason ? 'bg-red-600 hover:bg-red-700 text-white shadow-red-200 active:scale-95' : 'bg-gray-300 text-gray-500 cursor-not-allowed']"
        >
          Confirm Freeze
        </button>
      </div>

    </div>
  </div>
</template>

<script>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'CARD_FREEZE_FORM',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const reason = ref('')

    const updateReason = (e) => {
      // FSM logic requires length > 0 for button to work (precondition),
      // and op: set for effect.
      reason.value = e.target.value
      signatureStore.freeze_reason = e.target.value
    }

    const goBack = () => {
      signatureStore.setCurrentPageId('CARD_DETAIL')
      router.push({ name: 'CARD_DETAIL' })
    }

    const confirmFreeze = () => {
      // Precondition check: reason length > 0
      if (!reason.value) return
      
      signatureStore.setCurrentPageId('CARD_FREEZE_SUCCESS')
      router.push({ name: 'CARD_FREEZE_SUCCESS' })
    }

    return {
      reason,
      updateReason,
      goBack,
      confirmFreeze
    }
  }
}
</script>