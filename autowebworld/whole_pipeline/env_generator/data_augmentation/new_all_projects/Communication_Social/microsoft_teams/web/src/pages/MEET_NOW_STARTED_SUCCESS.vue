<template>
  <div class="h-screen flex flex-col bg-gray-900 text-white items-center justify-center">
    <div class="bg-gray-800 rounded-xl shadow-2xl p-8 max-w-md w-full text-center border border-gray-700">
      <div class="w-20 h-20 bg-[#6264A7] rounded-full flex items-center justify-center mx-auto mb-6 animate-pulse">
         <svg xmlns="http://www.w3.org/2000/svg" class="h-10 w-10 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
      </div>
      
      <h2 class="text-2xl font-bold mb-2">Meeting Started</h2>
      <p class="text-gray-400 mb-8">{{ store.success_message || 'You are now in the meeting.' }}</p>
      
      <button 
        id="meet-now-success-go-home"
        @click="goHome"
        class="w-full bg-white text-gray-900 hover:bg-gray-100 font-semibold py-3 px-6 rounded-lg shadow transition-colors"
      >
        Leave & Return Home
      </button>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'MEET_NOW_STARTED_SUCCESS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const goHome = async () => {
      store.success_message = "Meet now started";
      store.currentPageId = 'HOME';
      await router.push({ name: 'HOME' });
    }

    return {
      store,
      goHome
    }
  }
}
</script>