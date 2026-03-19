<template>
  <div class="min-h-screen bg-black text-white font-sans flex flex-col items-center justify-center p-4">
    <div class="text-center max-w-lg space-y-8 animate-fade-in-up">
       <div class="w-24 h-24 bg-[#1DB954] rounded-full flex items-center justify-center mx-auto shadow-[0_0_40px_rgba(29,185,84,0.5)]">
          <svg class="w-12 h-12 text-black" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7"/></svg>
       </div>
       
       <h1 class="text-4xl font-bold tracking-tight">Shared Successfully!</h1>
       <p class="text-xl text-[#B3B3B3]">Your friends are going to love this.</p>
       
       <button 
          id="success-go-home" 
          @click="handleGoHome" 
          class="bg-white text-black font-bold py-4 px-12 rounded-full hover:scale-105 transition-transform uppercase tracking-widest text-sm shadow-lg"
       >
          Return Home
       </button>
    </div>
  </div>
</template>

<script>
import { useSignatureStore } from '../stores/signature'
import { useRouter } from 'vue-router'

export default {
  name: 'PLAYLIST_SHARED_SUCCESS',
  setup() {
    const store = useSignatureStore()
    const router = useRouter()

    const handleGoHome = async () => {
       store.setCurrentPageId('HOME')
       await router.push({ name: 'HOME' })
    }

    return {
       handleGoHome
    }
  }
}
</script>

<style scoped>
@keyframes fade-in-up {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}
.animate-fade-in-up { animation: fade-in-up 0.8s ease-out forwards; }
</style>