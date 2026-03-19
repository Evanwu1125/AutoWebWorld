<template>
  <div class="min-h-screen bg-slate-50 flex items-center justify-center font-inter text-slate-900 px-4">
    <div class="bg-white p-8 rounded-lg shadow-lg border border-slate-200 max-w-md w-full text-center">
       <div class="mx-auto flex items-center justify-center h-16 w-16 rounded-full bg-green-100 mb-6">
          <span class="text-3xl">🔗</span>
       </div>
       <h2 class="text-2xl font-bold text-slate-900 mb-2">Merged!</h2>
       <p class="text-slate-600 mb-8">{{ signatureStore.success_message || 'Tickets merged successfully.' }}</p>
       
       <div class="space-y-3">
          <button id="merge-success-back-ticket" 
                  @click="handleBackToTicket" 
                  class="w-full bg-white border border-slate-300 text-slate-700 hover:bg-slate-50 font-medium py-3 px-4 rounded-md shadow-sm transition-colors duration-200">
            Back to Ticket #{{ currentTicketId }}
          </button>
          <button id="merge-success-go-home" 
                  @click="handleGoHome" 
                  class="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 px-4 rounded-md shadow-sm transition-colors duration-200">
            Go to Home
          </button>
       </div>
    </div>
  </div>
</template>

<script>
import { onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'MERGE_SUCCESS',
  setup() {
    const router = useRouter()
    const route = useRoute()
    const signatureStore = useSignatureStore()
    const currentTicketId = route.params.id

    onMounted(() => {
        signatureStore.success_message = "Tickets merged successfully"
    })

    const handleBackToTicket = async () => {
        signatureStore.setCurrentPageId('TICKET_DETAIL')
        await router.push({ name: 'TICKET_DETAIL', params: { id: currentTicketId } })
    }

    const handleGoHome = async () => {
        signatureStore.setCurrentPageId('HOME')
        await router.push({ name: 'HOME' })
    }

    return {
        signatureStore,
        currentTicketId,
        handleBackToTicket,
        handleGoHome
    }
  }
}
</script>