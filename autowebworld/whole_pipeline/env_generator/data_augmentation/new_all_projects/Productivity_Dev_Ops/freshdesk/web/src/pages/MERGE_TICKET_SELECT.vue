<template>
  <div class="min-h-screen bg-slate-50 font-inter text-slate-900 flex flex-col">
    <header class="bg-white shadow-sm z-20">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center">
         <h1 class="text-xl font-bold text-slate-900">Merge Ticket</h1>
      </div>
    </header>

    <main class="flex-1 max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
       <div class="bg-white p-6 rounded-lg shadow-sm border border-slate-200 mb-6">
          <p class="text-sm text-slate-500 mb-4">Search for a ticket to merge with <strong>#{{ currentTicketId }}</strong>.</p>
          
          <!-- Search -->
          <div class="relative rounded-md shadow-sm mb-6">
             <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <span class="text-slate-400">🔍</span>
             </div>
             <input type="text" 
                    id="merge-search-input"
                    v-model="searchQuery"
                    @keypress.enter="handleSearch"
                    class="focus:ring-blue-500 focus:border-blue-500 block w-full pl-10 sm:text-sm border-slate-300 rounded-md py-2" 
                    placeholder="Search by ID or Subject...">
          </div>

          <!-- Results List -->
          <div class="bg-white shadow overflow-hidden sm:rounded-md border border-slate-200" id="merge-results">
            <ul role="list" class="divide-y divide-slate-200 max-h-96 overflow-y-auto">
               <li v-for="ticket in filteredTickets" :key="ticket.id" class="hover:bg-slate-50 transition-colors duration-150">
                  <div 
                     :class="[
                       'block px-4 py-4 sm:px-6 cursor-pointer',

                       isMatched(ticket) ? 'row-matched' : '',
                       'row-visible',
                       'data-id-' + ticket.id,
                       selectedMergeId === ticket.id ? 'bg-blue-50 ring-2 ring-inset ring-blue-500' : ''
                     ]"
                     @click="handleSelect(ticket)"
                  >
                     <div class="flex items-center justify-between">
                        <div class="flex items-center truncate">
                           <p class="text-sm font-medium text-blue-600 truncate mr-4">#{{ ticket.id }}</p>
                           <p class="text-sm text-slate-900 truncate">{{ ticket.subject }}</p>
                        </div>
                        <div class="ml-2 flex-shrink-0 flex">
                           <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-slate-100 text-slate-800">
                              {{ ticket.status }}
                           </span>
                        </div>
                     </div>
                  </div>
               </li>
               <li v-if="filteredTickets.length === 0" class="px-4 py-8 text-center text-slate-500">
                  No tickets found.
               </li>
            </ul>
          </div>
       </div>

       <div class="flex justify-between">
          <button id="merge-select-back-ticket" 
                  @click="handleBack" 
                  class="bg-white border border-slate-300 text-slate-700 hover:bg-slate-50 font-medium py-2 px-4 rounded-md shadow-sm transition-colors">
             Cancel
          </button>
          <button id="btn-merge-next" 
                  @click="handleNext"
                  :disabled="!selectedMergeId"
                  :class="[
                     'px-6 py-2 rounded-md font-medium text-sm transition-colors shadow-sm',
                     selectedMergeId ? 'bg-blue-600 text-white hover:bg-blue-700' : 'bg-slate-200 text-slate-400 cursor-not-allowed'
                  ]">
             Next
          </button>
       </div>
    </main>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'MERGE_TICKET_SELECT',
  setup() {
    const router = useRouter()
    const route = useRoute()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()
    
    const currentTicketId = route.params.id
    const searchQuery = ref('')
    const selectedMergeId = ref(null)

    const filteredTickets = computed(() => {
        let result = dataStore.tickets.filter(t => t.id !== currentTicketId) // Exclude current
        if (searchQuery.value) {
            const q = searchQuery.value.toLowerCase()
            result = result.filter(t => 
                t.subject.toLowerCase().includes(q) || 
                t.id.toLowerCase().includes(q)
            )
        }
        return result
    })

    const isMatched = (ticket) => {
        if (!signatureStore.merge_search_has_searched) return false
        if (filteredTickets.value.length > 0 && filteredTickets.value[0].id === ticket.id) return true
        return false
    }

    const handleSearch = () => {
        signatureStore.merge_search_has_searched = true
        signatureStore.matched_merge_ticket_id = filteredTickets.value.length > 0 ? filteredTickets.value[0].id : null
    }

    const handleSelect = (ticket) => {
        selectedMergeId.value = ticket.id
        signatureStore.ticket_merge_target_id = ticket.id
        
        // Reset search flags if manually selected
        if (signatureStore.merge_search_has_searched) signatureStore.merge_search_has_searched = null
    }

    const handleBack = async () => {
        signatureStore.setCurrentPageId('TICKET_DETAIL')
        await router.push({ name: 'TICKET_DETAIL', params: { id: currentTicketId } })
    }

    const handleNext = async () => {
        if (!selectedMergeId.value) return
        signatureStore.setCurrentPageId('MERGE_TICKET_CONFIRM')
        await router.push({ name: 'MERGE_TICKET_CONFIRM', params: { id: currentTicketId } })
    }

    return {
        currentTicketId,
        searchQuery,
        filteredTickets,
        selectedMergeId,
        handleSearch,
        handleSelect,
        handleBack,
        handleNext,
        isMatched
    }
  }
}
</script>