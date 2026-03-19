<template>
  <div class="min-h-screen bg-[#0F0F0F] text-white flex flex-col items-center justify-center p-6">
    <div class="bg-[#1F1F1F] rounded-2xl p-8 max-w-md w-full text-center shadow-2xl border border-gray-800 transform transition-all animate-fade-in-up">
      <div class="w-20 h-20 bg-blue-500/20 text-blue-500 rounded-full flex items-center justify-center mx-auto mb-6">
        <svg class="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path></svg>
      </div>
      
      <h1 class="text-2xl font-bold mb-4">Video Published!</h1>
      <p class="text-gray-400 mb-8">
        Your video has been successfully uploaded and processed.
      </p>

      <div class="bg-[#121212] rounded-lg p-4 mb-8 flex gap-4 text-left">
         <div class="w-24 aspect-video bg-gray-800 rounded overflow-hidden">
            <div class="w-full h-full bg-blue-900/50 flex items-center justify-center text-xs">Thumbnail</div>
         </div>
         <div class="flex-1 min-w-0">
            <div class="text-sm font-bold truncate text-gray-200">my_awesome_video.mp4</div>
            <div class="text-xs text-blue-400 truncate mt-1">https://youtu.be/dQw4w9WgXcQ</div>
         </div>
      </div>
      
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
  name: 'UPLOAD_PUBLISH_SUCCESS',
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