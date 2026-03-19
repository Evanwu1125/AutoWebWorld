<template>
  <div class="min-h-screen bg-[#0F0F0F] text-white flex flex-col items-center justify-center p-6">
    <div class="bg-[#1F1F1F] rounded-2xl p-8 max-w-md w-full text-center shadow-2xl border border-gray-800 transform transition-all animate-fade-in-up">
      <div class="w-20 h-20 bg-red-500/20 text-red-500 rounded-full flex items-center justify-center mx-auto mb-6">
        <svg class="w-10 h-10" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg>
      </div>
      
      <h1 class="text-2xl font-bold mb-4">Subscribed!</h1>
      <p class="text-gray-400 mb-8">
        You are now subscribed to this channel.
      </p>
      
      <button 
        id="success-go-home"
        @click="goHome"
        class="w-full bg-[#FF0000] hover:bg-red-600 text-white font-bold py-3 px-6 rounded-full transition-colors"
      >
        Return Home
      </button>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'SUBSCRIBE_SUCCESS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const goHome = () => {
      store.currentPageId = 'HOME'
      router.push({ name: 'HOME' })
    }

    return { goHome }
  }
}
</script>