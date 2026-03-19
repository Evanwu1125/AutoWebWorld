<template>
  <div class="min-h-screen bg-slate-900 flex flex-col items-center justify-center p-4 text-center">
    <div class="mb-8">
      <div class="w-24 h-24 bg-purple-500 rounded-full flex items-center justify-center mx-auto mb-6 shadow-2xl shadow-purple-500/30 animate-pulse">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-12 w-12 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
      </div>
      <h1 class="text-3xl font-bold text-white mb-2">Scheduled!</h1>
      <p class="text-slate-400 text-lg">{{ store.success_message || 'Your post is in the queue.' }}</p>
    </div>

    <div class="flex flex-col sm:flex-row gap-4 w-full max-w-sm">
      <button 
        id="schedule-success-go-home" 
        @click="goHome"
        class="flex-1 py-3 px-6 rounded-full font-bold bg-slate-800 text-white hover:bg-slate-700 border border-slate-700 transition-colors"
      >
        Go Home
      </button>
      <button 
        id="schedule-success-back-dashboard" 
        @click="goDashboard"
        class="flex-1 py-3 px-6 rounded-full font-bold bg-white text-slate-900 hover:bg-slate-100 shadow-lg transition-colors"
      >
        Dashboard
      </button>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'POST_SCHEDULE_SUCCESS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const goHome = async () => {
      store.currentPageId = 'HOME'
      await router.push({ name: 'HOME' })
    }

    const goDashboard = async () => {
      store.currentPageId = 'DASHBOARD_FEED'
      await router.push({ name: 'DASHBOARD_FEED' })
    }

    return {
      store,
      goHome,
      goDashboard
    }
  }
}
</script>