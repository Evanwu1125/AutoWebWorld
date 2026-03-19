<template>
  <div class="min-h-screen bg-blue-600 flex flex-col items-center justify-center p-6 text-white text-center">
    
    <div class="w-24 h-24 bg-white rounded-full flex items-center justify-center mb-8 shadow-2xl animate-bounce-in">
      <svg class="w-12 h-12 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7"></path></svg>
    </div>

    <h1 class="text-3xl font-extrabold mb-2">Limits Updated</h1>
    <p class="text-blue-100 text-lg mb-12">Your card spending limits have been successfully saved.</p>

    <div class="w-full max-w-xs space-y-4">
      <button 
        id="btn-limits-go-home"
        @click="goHome"
        class="w-full py-4 bg-white text-blue-600 font-bold rounded-xl shadow-lg hover:bg-gray-50 transition-all active:scale-95"
      >
        Go Home
      </button>
      
      <button 
        id="btn-back-card-from-limits"
        @click="goBackToCard"
        class="w-full py-4 bg-blue-700 text-white font-bold rounded-xl hover:bg-blue-800 transition-all border border-blue-500"
      >
        Return to Card
      </button>
    </div>

  </div>
</template>

<script>
import { onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'CARD_LIMITS_SUCCESS',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    onMounted(() => {
      signatureStore.success_message = "Card limits updated"
    })

    const goHome = () => {
      signatureStore.setCurrentPageId('HOME')
      router.push({ name: 'HOME' })
    }

    const goBackToCard = () => {
      signatureStore.setCurrentPageId('CARD_DETAIL')
      router.push({ name: 'CARD_DETAIL' })
    }

    return {
      goHome,
      goBackToCard
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