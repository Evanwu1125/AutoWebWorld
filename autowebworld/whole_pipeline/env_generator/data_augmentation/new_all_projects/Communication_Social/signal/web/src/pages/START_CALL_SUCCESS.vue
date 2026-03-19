<template>
  <div class="h-screen bg-slate-50 flex flex-col items-center justify-center p-4">
    <div class="bg-white w-full max-w-sm rounded-2xl shadow-xl p-8 text-center space-y-6">
        <div class="w-20 h-20 bg-green-100 text-green-500 rounded-full flex items-center justify-center mx-auto animate-pulse">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-10 w-10" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
            </svg>
        </div>
        
        <div>
            <h2 class="text-2xl font-bold text-slate-800">Calling...</h2>
            <p class="text-slate-500 mt-2">Connecting securely.</p>
        </div>

        <div class="space-y-3 pt-4">
            <button 
                id="call-success-back-history" 
                @click="goHistory"
                class="w-full py-3 px-4 bg-red-100 text-red-600 font-semibold rounded-xl hover:bg-red-200 transition-colors"
            >
                End Call
            </button>
            <button 
                id="call-success-go-home" 
                @click="goHome"
                class="w-full py-3 px-4 bg-slate-100 text-slate-700 font-semibold rounded-xl hover:bg-slate-200 transition-colors"
            >
                Go Home
            </button>
        </div>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'START_CALL_SUCCESS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const goHistory = async () => {
        store.currentPageId = 'CALL_HISTORY'
        await router.push({ name: 'CALL_HISTORY' })
    }

    const goHome = async () => {
        store.currentPageId = 'HOME'
        await router.push({ name: 'HOME' })
    }

    return {
        goHistory,
        goHome
    }
  }
}
</script>