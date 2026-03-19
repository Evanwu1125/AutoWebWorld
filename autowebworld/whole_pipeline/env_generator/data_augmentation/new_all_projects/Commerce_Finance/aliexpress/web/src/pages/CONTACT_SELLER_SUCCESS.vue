<template>
  <div class="min-h-screen bg-gray-50 flex flex-col items-center justify-center p-4 text-center">
    <div class="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center mb-6">
      <svg class="w-10 h-10 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"></path></svg>
    </div>
    
    <h1 class="text-2xl font-bold text-gray-900 mb-2">Message Sent!</h1>
    <p class="text-gray-600 mb-8">The seller has received your message and will respond shortly.</p>
    
    <button 
      id="contact-success-go-home" 
      class="bg-red-600 text-white font-bold py-3 px-8 rounded-full shadow-lg hover:bg-red-700 transition-colors"
      @click="handleGoHome"
    >
      Return to Home
    </button>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'CONTACT_SELLER_SUCCESS',
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