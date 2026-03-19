<template>
  <div class="min-h-screen bg-gray-50 flex flex-col items-center justify-center p-4 text-center">
    <div class="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center mb-6">
      <span class="text-4xl">💳</span>
    </div>
    
    <h1 class="text-2xl font-bold text-gray-900 mb-2">Card Added!</h1>
    <p class="text-gray-600 mb-8">Your card has been successfully saved for future payments.</p>
    
    <button 
      id="card-success-go-home" 
      class="bg-red-600 text-white font-bold py-3 px-8 rounded-full shadow-lg hover:bg-red-700 transition-colors"
      @click="handleGoHome"
    >
      Done
    </button>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'ORDER_CARD_SUCCESS',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const handleGoHome = async () => {
       signatureStore.currentPageId = 'HOME'
       await router.push({ name: 'HOME' })
    }

    return {
       handleGoHome
    }
  }
}
</script>