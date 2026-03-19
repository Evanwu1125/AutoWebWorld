<template>
  <div class="min-h-screen bg-black text-white font-sans flex flex-col items-center justify-center p-4">
    <div class="text-center max-w-lg space-y-8 animate-fade-in-up">
       <!-- Success Animation -->
       <div class="relative w-32 h-32 mx-auto">
          <div class="absolute inset-0 bg-[#1DB954] rounded-full opacity-20 animate-pulse"></div>
          <div class="absolute inset-4 bg-[#1DB954] rounded-full flex items-center justify-center shadow-[0_0_50px_rgba(29,185,84,0.6)]">
             <svg class="w-16 h-16 text-black" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7"/></svg>
          </div>
       </div>
       
       <h1 class="text-5xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-[#1DB954]">Premium Unlocked</h1>
       <p class="text-xl text-[#B3B3B3]">Enjoy ad-free music, offline playback, and unlimited skips.</p>
       
       <button 
          id="success-go-home" 
          @click="handleGoHome" 
          class="bg-white text-black font-bold py-4 px-12 rounded-full hover:scale-105 transition-transform uppercase tracking-widest text-sm shadow-lg"
       >
          Start Listening
       </button>
    </div>
  </div>
</template>

<script>
import { useSignatureStore } from '../stores/signature'
import { useRouter } from 'vue-router'

export default {
  name: 'PREMIUM_UPGRADE_SUCCESS',
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