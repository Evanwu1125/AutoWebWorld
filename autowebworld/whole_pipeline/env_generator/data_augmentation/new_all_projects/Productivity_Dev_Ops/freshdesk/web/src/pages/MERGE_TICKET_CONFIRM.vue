<template>
  <div class="min-h-screen bg-slate-50 font-inter text-slate-900 flex flex-col">
    <header class="bg-white shadow-sm z-20">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center">
         <h1 class="text-xl font-bold text-slate-900">Confirm Merge</h1>
      </div>
    </header>

    <main class="flex-1 max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
       <div class="bg-white p-8 rounded-lg shadow-sm border border-slate-200 text-center space-y-6">
          <div class="flex items-center justify-center space-x-8 mb-8">
             <div class="bg-slate-100 p-4 rounded-lg">
                <span class="block text-2xl font-bold text-blue-600">#{{ currentTicketId }}</span>
                <span class="text-sm text-slate-500">Primary</span>
             </div>
             <div class="text-slate-400 text-xl">⬅️ merges with</div>
             <div class="bg-slate-100 p-4 rounded-lg">
                <span class="block text-2xl font-bold text-slate-700">#{{ targetId }}</span>
                <span class="text-sm text-slate-500">Secondary (will close)</span>
             </div>
          </div>
          
          <p class="text-slate-600 max-w-md mx-auto">
             Are you sure you want to merge these tickets? The secondary ticket will be closed and all its conversations will be moved to the primary ticket. This action cannot be undone.
          </p>

          <div class="flex justify-center space-x-4 pt-4">
             <button id="merge-confirm-back-select" 
                     @click="handleBack" 
                     class="bg-white border border-slate-300 text-slate-700 hover:bg-slate-50 font-medium py-2 px-6 rounded-md shadow-sm transition-colors">
                Cancel
             </button>
             <button id="btn-merge-confirm" 
                     @click="handleConfirm" 
                     class="bg-red-600 hover:bg-red-700 text-white font-medium py-2 px-6 rounded-md shadow-sm transition-colors">
                Confirm Merge
             </button>
          </div>
       </div>
    </main>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'MERGE_TICKET_CONFIRM',
  setup() {
    const router = useRouter()
    const route = useRoute()
    const signatureStore = useSignatureStore()
    
    const currentTicketId = route.params.id
    const targetId = computed(() => signatureStore.ticket_merge_target_id)

    const handleBack = async () => {
        signatureStore.setCurrentPageId('MERGE_TICKET_SELECT')
        await router.push({ name: 'MERGE_TICKET_SELECT', params: { id: currentTicketId } })
    }

    const handleConfirm = async () => {
        // Logic to actually merge would go here (update store/mock data)
        signatureStore.setCurrentPageId('MERGE_SUCCESS')
        await router.push({ name: 'MERGE_SUCCESS', params: { id: currentTicketId } })
    }

    return {
        currentTicketId,
        targetId,
        handleBack,
        handleConfirm
    }
  }
}
</script>