<template>
  <div class="min-h-screen bg-[#0F0F0F] text-white flex flex-col items-center justify-center p-6">
    <div class="bg-[#1F1F1F] rounded-2xl p-8 max-w-md w-full text-center shadow-2xl border border-gray-800 transform transition-all animate-fade-in-up">
      <div class="w-20 h-20 bg-blue-500/20 text-blue-500 rounded-full flex items-center justify-center mx-auto mb-6">
        <svg class="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"></path></svg>
      </div>
      
      <h1 class="text-2xl font-bold mb-4">Comment Posted</h1>
      <p class="text-gray-400 mb-8">
        Your comment has been successfully published to the video.
      </p>
      
      <button 
        id="success-go-home"
        @click="goHome"
        class="w-full bg-[#3EA6FF] hover:bg-blue-600 text-black hover:text-white font-bold py-3 px-6 rounded-full transition-colors"
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
  name: 'WATCH_COMMENT_SUCCESS',
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