<template>
  <div class="min-h-screen bg-slate-50 font-inter text-slate-900 flex flex-col">
    <header class="bg-white shadow-sm z-20">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center">
         <h1 class="text-xl font-bold text-slate-900">Assign Ticket</h1>
      </div>
    </header>

    <main class="flex-1 max-w-lg mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
       <div class="bg-white p-8 rounded-lg shadow-sm border border-slate-200 space-y-6">
          <p class="text-sm text-slate-500">Select an agent to assign this ticket to.</p>
          
          <div class="relative">
             <label class="block text-sm font-medium text-slate-700 mb-1">Agent</label>
             <div class="relative">
                <button id="assign-agent-dropdown" @click="toggleDropdown" class="w-full bg-white border border-slate-300 rounded-md py-2 px-3 text-left shadow-sm focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 sm:text-sm">
                   <span class="block truncate">{{ selectedAgent || 'Select Agent' }}</span>
                   <span class="absolute inset-y-0 right-0 flex items-center pr-2 pointer-events-none text-slate-400">▼</span>
                </button>
                <div v-if="dropdownOpen" class="absolute z-10 mt-1 w-full bg-white shadow-lg max-h-60 rounded-md py-1 text-base ring-1 ring-black ring-opacity-5 overflow-auto focus:outline-none sm:text-sm">
                   <div id="agent-1" @click="handleSelectAgent('agent1')" class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-blue-50 text-slate-900 flex items-center">
                      <img src="/images/photo1765352638.jpg" class="h-6 w-6 rounded-full mr-2"> Agent 1
                   </div>
                   <div id="agent-2" @click="handleSelectAgent('agent2')" class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-blue-50 text-slate-900 flex items-center">
                      <img src="/images/Agent2.jpg" class="h-6 w-6 rounded-full mr-2"> Agent 2
                   </div>
                   <div id="agent-any" @click="handleSelectAgent('agent_any')" class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-blue-50 text-slate-900 flex items-center">
                      <span class="h-6 w-6 rounded-full bg-slate-200 mr-2 flex items-center justify-center text-xs">?</span> Any Agent
                   </div>
                </div>
             </div>
          </div>

          <div class="flex justify-between pt-4">
             <button id="back-ticket-detail-from-assign" 
                     @click="handleBack" 
                     class="bg-white border border-slate-300 text-slate-700 hover:bg-slate-50 font-medium py-2 px-4 rounded-md shadow-sm transition-colors">
                Cancel
             </button>
             <button id="btn-assign-confirm" 
                     @click="handleConfirm"
                     :disabled="!selectedAgent"
                     :class="[
                        'px-6 py-2 rounded-md font-medium text-sm transition-colors shadow-sm',
                        selectedAgent ? 'bg-blue-600 text-white hover:bg-blue-700' : 'bg-slate-200 text-slate-400 cursor-not-allowed'
                     ]">
                Confirm Assignment
             </button>
          </div>
       </div>
    </main>
  </div>
</template>

<script>
import { ref } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'ASSIGN_TICKET',
  setup() {
    const router = useRouter()
    const route = useRoute()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()
    const ticketId = route.params.id

    const selectedAgent = ref('')
    const dropdownOpen = ref(false)

    const toggleDropdown = () => dropdownOpen.value = !dropdownOpen.value

    const handleSelectAgent = (agentId) => {
        selectedAgent.value = agentId
        signatureStore.ticket_assignee_id = agentId
        dropdownOpen.value = false
    }

    const handleBack = async () => {
        signatureStore.setCurrentPageId('TICKET_DETAIL')
        await router.push({ name: 'TICKET_DETAIL', params: { id: ticketId } })
    }

    const handleConfirm = async () => {
        if (!selectedAgent.value) return
        
        // Update mock data
        const ticket = dataStore.getTicketById(ticketId)
        if (ticket) {
            ticket.assignee_id = selectedAgent.value
        }

        signatureStore.setCurrentPageId('ASSIGN_SUCCESS')
        await router.push({ name: 'ASSIGN_SUCCESS', params: { id: ticketId } })
    }

    return {
        selectedAgent,
        dropdownOpen,
        toggleDropdown,
        handleSelectAgent,
        handleBack,
        handleConfirm
    }
  }
}
</script>