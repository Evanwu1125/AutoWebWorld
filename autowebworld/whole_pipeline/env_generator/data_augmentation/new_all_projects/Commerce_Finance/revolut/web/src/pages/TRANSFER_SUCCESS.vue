<template>
  <div class="min-h-screen bg-blue-600 flex flex-col items-center justify-center p-6 text-white text-center">
    
    <div class="w-24 h-24 bg-white rounded-full flex items-center justify-center mb-8 shadow-2xl animate-bounce-in">
      <svg class="w-12 h-12 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7"></path></svg>
    </div>

    <h1 class="text-3xl font-extrabold mb-2">Transfer Sent!</h1>
    <p class="text-blue-100 text-lg mb-12 max-w-xs">Your money is on its way to {{ beneficiaryName }}.</p>

    <div class="w-full max-w-xs space-y-4">
      <button 
        id="btn-go-home"
        @click="goHome"
        class="w-full py-4 bg-white text-blue-600 font-bold rounded-xl shadow-lg hover:bg-gray-50 transition-all active:scale-95"
      >
        Go Home
      </button>
      
      <button 
        id="btn-view-details"
        @click="goBackToReview"
        class="w-full py-4 bg-blue-700 text-white font-bold rounded-xl hover:bg-blue-800 transition-all border border-blue-500"
      >
        View Details
      </button>
    </div>

  </div>
</template>

<script>
import { computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'TRANSFER_SUCCESS',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const beneficiaryName = computed(() => {
      const b = dataStore.beneficiaries.find(i => i.id === signatureStore.payments_selected_beneficiary_id)
      return b ? b.name : 'Recipient'
    })

    onMounted(() => {
      // Set success message effect
      signatureStore.success_message = "Transfer completed"
    })

    const goHome = () => {
      signatureStore.setCurrentPageId('HOME')
      router.push({ name: 'HOME' })
    }

    const goBackToReview = () => {
      signatureStore.setCurrentPageId('TRANSFER_REVIEW')
      router.push({ name: 'TRANSFER_REVIEW' })
    }

    return {
      beneficiaryName,
      goHome,
      goBackToReview
    }
  }
}
</script>

<style scoped>
.animate-bounce-in {
  animation: bounceIn 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55);
}

@keyframes bounceIn {
  0% { transform: scale(0); opacity: 0; }
  100% { transform: scale(1); opacity: 1; }
}
</style>