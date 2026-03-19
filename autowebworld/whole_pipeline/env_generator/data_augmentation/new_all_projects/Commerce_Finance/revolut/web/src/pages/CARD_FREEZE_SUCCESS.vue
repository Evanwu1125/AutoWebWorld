<template>
  <div class="min-h-screen bg-gray-900 flex flex-col items-center justify-center p-6 text-white text-center">
    
    <div class="w-24 h-24 bg-gray-800 rounded-full flex items-center justify-center mb-8 shadow-2xl animate-fade-in border-4 border-gray-700">
      <svg class="w-10 h-10 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"></path></svg>
    </div>

    <h1 class="text-3xl font-extrabold mb-2">Card Frozen</h1>
    <p class="text-gray-400 text-lg mb-12">Your card is now secure and inactive.</p>

    <div class="w-full max-w-xs space-y-4">
      <button 
        id="btn-freeze-go-home"
        @click="goHome"
        class="w-full py-4 bg-white text-gray-900 font-bold rounded-xl shadow-lg hover:bg-gray-200 transition-all active:scale-95"
      >
        Go Home
      </button>
      
      <button 
        id="btn-back-to-card"
        @click="goBackToCard"
        class="w-full py-4 bg-gray-800 text-white font-bold rounded-xl hover:bg-gray-700 transition-all border border-gray-600"
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
  name: 'CARD_FREEZE_SUCCESS',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    onMounted(() => {
      signatureStore.success_message = "Card frozen"
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
.animate-fade-in {
  animation: fadeIn 0.8s ease-out;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>