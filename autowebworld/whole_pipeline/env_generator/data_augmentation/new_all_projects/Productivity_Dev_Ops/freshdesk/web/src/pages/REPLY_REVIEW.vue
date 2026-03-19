<template>
  <div class="min-h-screen bg-slate-50 font-inter text-slate-900 flex flex-col">
    <header class="bg-white shadow-sm z-20">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center">
         <h1 class="text-xl font-bold text-slate-900">Review Reply</h1>
      </div>
    </header>

    <main class="flex-1 max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
       <div class="bg-white p-8 rounded-lg shadow-sm border border-slate-200 space-y-6">
          <div class="bg-blue-50 p-4 rounded-md border border-blue-100">
             <h3 class="text-xs font-bold text-blue-800 uppercase tracking-wider mb-2">Reply Content</h3>
             <p class="text-slate-800 whitespace-pre-wrap">Your draft reply content would appear here based on what you typed in the previous screen.</p>
          </div>

          <div class="flex justify-between pt-4">
             <button id="back-ticket-detail" 
                     @click="handleBack" 
                     class="bg-white border border-slate-300 text-slate-700 hover:bg-slate-50 font-medium py-2 px-4 rounded-md shadow-sm transition-colors">
                Edit Reply
             </button>
             <button id="btn-send-reply" 
                     @click="handleSend" 
                     class="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-6 rounded-md shadow-sm transition-colors">
                Send Reply
             </button>
          </div>
       </div>
    </main>
  </div>
</template>

<script>
import { useRouter, useRoute } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'REPLY_REVIEW',
  setup() {
    const router = useRouter()
    const route = useRoute()
    const signatureStore = useSignatureStore()
    const ticketId = route.params.id

    const handleBack = async () => {
        signatureStore.setCurrentPageId('TICKET_DETAIL')
        await router.push({ name: 'TICKET_DETAIL', params: { id: ticketId } })
    }

    const handleSend = async () => {
        signatureStore.setCurrentPageId('REPLY_SENT_SUCCESS')
        await router.push({ name: 'REPLY_SENT_SUCCESS', params: { id: ticketId } })
    }

    return {
        handleBack,
        handleSend
    }
  }
}
</script>