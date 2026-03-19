<template>
  <div class="min-h-screen bg-[#0F0F0F] text-white flex flex-col items-center justify-center p-6">
    <div class="bg-[#1F1F1F] rounded-2xl p-8 max-w-md w-full text-center shadow-2xl border border-gray-800 transform transition-all animate-fade-in-up">
      <div class="w-20 h-20 bg-green-500/20 text-green-500 rounded-full flex items-center justify-center mx-auto mb-6">
        <svg class="w-10 h-10" fill="currentColor" viewBox="0 0 24 24"><path d="M1 21h4V9H1v12zm22-11c0-1.1-.9-2-2-2h-6.31l.95-4.57.03-.32c0-.41-.17-.79-.44-1.06L14.17 1 7.59 7.59C7.22 7.95 7 8.45 7 9v10c0 1.1.9 2 2 2h9c.83 0 1.54-.5 1.84-1.22l3.02-7.05c.09-.23.14-.47.14-.73v-1.91l-.01-.01L23 10z"/></svg>
      </div>
      
      <h1 class="text-2xl font-bold mb-4">Video Liked!</h1>
      <p class="text-gray-400 mb-8">
        This video has been added to your Liked Videos playlist.
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
  name: 'WATCH_LIKE_SUCCESS',
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