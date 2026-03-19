<template>
  <div class="min-h-screen bg-[#0F0F0F] text-white flex flex-col items-center justify-center p-6">
    <div class="bg-[#1F1F1F] rounded-2xl p-8 max-w-md w-full text-center shadow-2xl border border-gray-800 transform transition-all animate-fade-in-up">
      <div class="w-20 h-20 bg-purple-500/20 text-purple-500 rounded-full flex items-center justify-center mx-auto mb-6">
        <svg class="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path></svg>
      </div>
      
      <h1 class="text-2xl font-bold mb-4">Playlist Created!</h1>
      <p class="text-gray-400 mb-8">
        Your new playlist is ready. You can now add videos to it.
      </p>
      
      <button 
        id="success-go-home"
        @click="goHome"
        class="w-full bg-[#3EA6FF] hover:bg-blue-600 text-black font-bold py-3 px-6 rounded-full transition-colors"
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
  name: 'PLAYLIST_CREATE_SUCCESS',
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