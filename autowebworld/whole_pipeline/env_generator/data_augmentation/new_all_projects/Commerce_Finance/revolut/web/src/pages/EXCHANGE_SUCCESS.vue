<template>
  <div class="min-h-screen bg-green-500 flex flex-col items-center justify-center p-6 text-white text-center">
    
    <div class="w-24 h-24 bg-white rounded-full flex items-center justify-center mb-8 shadow-2xl animate-scale-up">
      <svg class="w-12 h-12 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7"></path></svg>
    </div>

    <h1 class="text-3xl font-extrabold mb-2">Exchanged!</h1>
    <p class="text-green-100 text-lg mb-12">You've successfully exchanged your currency.</p>

    <div class="w-full max-w-xs space-y-4">
      <button 
        id="btn-exchange-go-home"
        @click="goHome"
        class="w-full py-4 bg-white text-green-600 font-bold rounded-xl shadow-lg hover:bg-gray-50 transition-all active:scale-95"
      >
        Go Home
      </button>
      
      <button 
        id="btn-view-exchange-details"
        @click="goBackToReview"
        class="w-full py-4 bg-green-600 text-white font-bold rounded-xl hover:bg-green-700 transition-all border border-green-400"
      >
        View Order
      </button>
    </div>

  </div>
</template>

<script>
import { onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'EXCHANGE_SUCCESS',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    onMounted(() => {
      signatureStore.success_message = "Exchange completed"
    })

    const goHome = () => {
      signatureStore.setCurrentPageId('HOME')
      router.push({ name: 'HOME' })
    }

    const goBackToReview = () => {
      signatureStore.setCurrentPageId('EXCHANGE_REVIEW')
      router.push({ name: 'EXCHANGE_REVIEW' })
    }

    return {
      goHome,
      goBackToReview
    }
  }
}
</script>

<style scoped>
.animate-scale-up {
  animation: scaleUp 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}

@keyframes scaleUp {
  0% { transform: scale(0); opacity: 0; }
  100% { transform: scale(1); opacity: 1; }
}
</style>