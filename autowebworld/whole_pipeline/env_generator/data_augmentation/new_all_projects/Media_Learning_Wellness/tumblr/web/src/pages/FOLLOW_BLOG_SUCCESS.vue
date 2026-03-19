<template>
  <div class="min-h-screen bg-slate-900 flex flex-col items-center justify-center p-4 text-center">
    <div class="mb-8">
      <div class="w-24 h-24 bg-green-500 rounded-full flex items-center justify-center mx-auto mb-6 shadow-2xl shadow-green-500/30 animate-bounce">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-12 w-12 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7" /></svg>
      </div>
      <h1 class="text-3xl font-bold text-white mb-2">Success!</h1>
      <p class="text-slate-400 text-lg">{{ store.success_message || 'Followed successfully.' }}</p>
    </div>

    <div class="flex flex-col sm:flex-row gap-4 w-full max-w-sm">
      <button 
        id="follow-success-go-home" 
        @click="goHome"
        class="flex-1 py-3 px-6 rounded-full font-bold bg-slate-800 text-white hover:bg-slate-700 border border-slate-700 transition-colors"
      >
        Go Home
      </button>
      <button 
        id="follow-success-back-overview" 
        @click="goBackOverview"
        class="flex-1 py-3 px-6 rounded-full font-bold bg-blue-500 text-white hover:bg-blue-600 shadow-lg shadow-blue-500/30 transition-colors"
      >
        Back to Blog
      </button>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'FOLLOW_BLOG_SUCCESS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const goHome = async () => {
      store.currentPageId = 'HOME'
      await router.push({ name: 'HOME' })
    }

    const goBackOverview = async () => {
      store.currentPageId = 'BLOG_OVERVIEW'
      // Need selected_blog_id to be present in store
      await router.push({ name: 'BLOG_OVERVIEW', params: { id: store.selected_blog_id } })
    }

    return {
      store,
      goHome,
      goBackOverview
    }
  }
}
</script>