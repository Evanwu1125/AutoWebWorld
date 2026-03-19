<template>
  <div class="min-h-screen bg-slate-900 flex flex-col items-center justify-center p-4 text-center">
    <div class="mb-8">
      <div class="w-20 h-20 bg-blue-500 rounded-full flex items-center justify-center mx-auto mb-6 shadow-2xl shadow-blue-500/30">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-10 w-10 text-white transform -rotate-45 ml-2" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" /></svg>
      </div>
      <h1 class="text-3xl font-bold text-white mb-2">Sent!</h1>
      <p class="text-slate-400 text-lg">{{ store.success_message || 'Message delivered.' }}</p>
    </div>

    <div class="flex flex-col sm:flex-row gap-4 w-full max-w-sm">
      <button 
        id="message-success-go-home" 
        @click="goHome"
        class="flex-1 py-3 px-6 rounded-full font-bold bg-slate-800 text-white hover:bg-slate-700 border border-slate-700 transition-colors"
      >
        Go Home
      </button>
      <button 
        id="message-success-back-inbox" 
        @click="goInbox"
        class="flex-1 py-3 px-6 rounded-full font-bold bg-white text-slate-900 hover:bg-slate-100 shadow-lg transition-colors"
      >
        Inbox
      </button>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'MESSAGE_SEND_SUCCESS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const goHome = async () => {
      store.currentPageId = 'HOME'
      await router.push({ name: 'HOME' })
    }

    const goInbox = async () => {
      store.currentPageId = 'MESSAGES_INBOX'
      await router.push({ name: 'MESSAGES_INBOX' })
    }

    return {
      store,
      goHome,
      goInbox
    }
  }
}
</script>