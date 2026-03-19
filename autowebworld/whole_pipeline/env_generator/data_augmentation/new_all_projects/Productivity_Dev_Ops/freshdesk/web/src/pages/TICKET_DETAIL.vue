<template>
  <div class="min-h-screen bg-slate-50 font-inter text-slate-900 flex flex-col">
    <!-- Header -->
    <header class="bg-white shadow-sm z-20">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
         <div class="flex items-center">
            <button id="back-tickets-list" @click="handleBackToList" class="mr-4 text-slate-500 hover:text-blue-600 transition-colors">
               ← Back
            </button>
            <h1 class="text-xl font-bold text-slate-900 truncate max-w-xl">
               <span class="text-slate-400 mr-2">#{{ ticket?.id }}</span> {{ ticket?.subject }}
            </h1>
         </div>
         <div class="flex space-x-3">
             <button id="btn-assign-ticket" @click="handleAssign" class="bg-white border border-slate-300 text-slate-700 hover:bg-slate-50 font-medium py-1.5 px-3 rounded text-sm transition-colors">
               Assign
             </button>
             <button id="btn-merge-ticket" @click="handleMerge" class="bg-white border border-slate-300 text-slate-700 hover:bg-slate-50 font-medium py-1.5 px-3 rounded text-sm transition-colors">
               Merge
             </button>
         </div>
      </div>
    </header>

    <main class="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full flex flex-col lg:flex-row gap-6">
       <!-- Main Content: Ticket Info & Reply -->
       <div class="flex-1 space-y-6">
          <!-- Ticket Description Card -->
          <div class="bg-white p-6 rounded-lg shadow-sm border border-slate-200">
             <div class="flex items-center mb-4">
               <img :src="ticket?.image" class="h-10 w-10 rounded-full mr-3" alt="Avatar">
               <div>
                  <div class="font-medium text-slate-900">Requester ({{ ticket?.requester_id }})</div>
                  <div class="text-xs text-slate-500">{{ formatDate(ticket?.created_at) }}</div>
               </div>
             </div>
             <div class="prose prose-slate max-w-none text-slate-800">
                <p>{{ ticket?.description }}</p>
             </div>
          </div>

          <!-- Reply Editor -->
          <div class="bg-white p-6 rounded-lg shadow-sm border border-slate-200">
             <h3 class="text-sm font-medium text-slate-900 mb-3">Reply to Customer</h3>
             <textarea id="reply-editor" 
                       v-model="replyText" 
                       @input="handleTypeReply"
                       class="w-full h-32 p-3 border border-slate-300 rounded-md focus:ring-blue-500 focus:border-blue-500 resize-none text-sm mb-4" 
                       placeholder="Type your reply here..."></textarea>
             <div class="flex justify-end">
                <button id="btn-review-reply" 
                        @click="handleReviewReply"
                        :disabled="!replyText"
                        :class="[
                          'px-4 py-2 rounded-md font-medium text-sm transition-colors shadow-sm',
                          replyText ? 'bg-blue-600 text-white hover:bg-blue-700' : 'bg-slate-200 text-slate-400 cursor-not-allowed'
                        ]">
                  Review & Send
                </button>
             </div>
          </div>
       </div>

       <!-- Sidebar: Properties -->
       <div class="w-full lg:w-80 space-y-6">
          <div class="bg-white p-6 rounded-lg shadow-sm border border-slate-200">
             <h3 class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-4">Properties</h3>
             
             <!-- Status Dropdown -->
             <div class="mb-4 relative">
                <label class="block text-sm font-medium text-slate-700 mb-1">Status</label>
                <div class="relative">
                   <button id="status-dropdown" @click="toggleStatusDropdown" class="w-full bg-white border border-slate-300 rounded-md py-2 px-3 text-left shadow-sm focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 sm:text-sm">
                      <span class="block truncate">{{ currentStatus }}</span>
                      <span class="absolute inset-y-0 right-0 flex items-center pr-2 pointer-events-none text-slate-400">▼</span>
                   </button>
                   <div v-if="statusDropdownOpen" class="absolute z-10 mt-1 w-full bg-white shadow-lg max-h-60 rounded-md py-1 text-base ring-1 ring-black ring-opacity-5 overflow-auto focus:outline-none sm:text-sm">
                      <div id="status-open" @click="handleSetStatus('Open')" class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-blue-50 text-slate-900">Open</div>
                      <div id="status-pending" @click="handleSetStatus('Pending')" class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-blue-50 text-slate-900">Pending</div>
                      <div id="status-resolved" @click="handleSetStatus('Resolved')" class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-blue-50 text-slate-900">Resolved</div>
                   </div>
                </div>
             </div>

             <!-- Priority Dropdown -->
             <div class="mb-4 relative">
                <label class="block text-sm font-medium text-slate-700 mb-1">Priority</label>
                <div class="relative">
                   <button id="priority-dropdown" @click="togglePriorityDropdown" class="w-full bg-white border border-slate-300 rounded-md py-2 px-3 text-left shadow-sm focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 sm:text-sm">
                      <span class="block truncate">{{ currentPriority }}</span>
                      <span class="absolute inset-y-0 right-0 flex items-center pr-2 pointer-events-none text-slate-400">▼</span>
                   </button>
                   <div v-if="priorityDropdownOpen" class="absolute z-10 mt-1 w-full bg-white shadow-lg max-h-60 rounded-md py-1 text-base ring-1 ring-black ring-opacity-5 overflow-auto focus:outline-none sm:text-sm">
                      <div id="priority-low" @click="handleSetPriority('Low')" class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-blue-50 text-slate-900">Low</div>
                      <div id="priority-medium" @click="handleSetPriority('Medium')" class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-blue-50 text-slate-900">Medium</div>
                      <div id="priority-high" @click="handleSetPriority('High')" class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-blue-50 text-slate-900">High</div>
                   </div>
                </div>
             </div>
             
             <!-- Read-only Agent info -->
             <div class="mb-4">
                <label class="block text-sm font-medium text-slate-700 mb-1">Agent</label>
                <div class="text-sm text-slate-900 bg-slate-100 px-3 py-2 rounded-md">
                   {{ ticket?.assignee_id || 'Unassigned' }}
                </div>
             </div>

          </div>
       </div>
    </main>
  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'TICKET_DETAIL',
  setup() {
    const route = useRoute()
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const ticketId = route.params.id
    const ticket = computed(() => dataStore.getTicketById(ticketId))

    const replyText = ref('')
    const statusDropdownOpen = ref(false)
    const priorityDropdownOpen = ref(false)
    
    // Local state to simulate updates until persisted to store or backend
    // In FSM, effects update store directly. We sync UI with store.
    const currentStatus = computed(() => signatureStore.ticket_status || ticket.value?.status || 'Open')
    const currentPriority = computed(() => signatureStore.ticket_priority || ticket.value?.priority || 'Medium')

    onMounted(() => {
       if (!ticket.value) {
          // Handle invalid ticket scenario if needed
       }
       // Initialize store state if not already set (for direct page load)
       signatureStore.selected_ticket_id = ticketId
    })

    const formatDate = (dateStr) => {
        if (!dateStr) return ''
        return new Date(dateStr).toLocaleString()
    }

    const handleBackToList = async () => {
        signatureStore.setCurrentPageId('TICKETS_LIST')
        await router.push({ name: 'TICKETS_LIST' })
    }

    const handleTypeReply = () => {
        if (replyText.value.length > 0) {
            signatureStore.ticket_reply_draft = 'has_content'
        } else {
            signatureStore.ticket_reply_draft = null
        }
    }

    const handleReviewReply = async () => {
        if (!replyText.value) return
        signatureStore.setCurrentPageId('REPLY_REVIEW')
        await router.push({ name: 'REPLY_REVIEW', params: { id: ticketId } })
    }

    const handleAssign = async () => {
        signatureStore.setCurrentPageId('ASSIGN_TICKET')
        await router.push({ name: 'ASSIGN_TICKET', params: { id: ticketId } })
    }

    const handleMerge = async () => {
        signatureStore.setCurrentPageId('MERGE_TICKET_SELECT')
        await router.push({ name: 'MERGE_TICKET_SELECT', params: { id: ticketId } })
    }

    const toggleStatusDropdown = () => statusDropdownOpen.value = !statusDropdownOpen.value
    const togglePriorityDropdown = () => priorityDropdownOpen.value = !priorityDropdownOpen.value

    const handleSetStatus = (status) => {
        signatureStore.ticket_status = status
        statusDropdownOpen.value = false
        // Optimistically update mock data for better UX
        if (ticket.value) ticket.value.status = status
    }

    const handleSetPriority = (priority) => {
        signatureStore.ticket_priority = priority
        priorityDropdownOpen.value = false
        // Optimistically update mock data
        if (ticket.value) ticket.value.priority = priority
    }

    return {
        ticket,
        replyText,
        statusDropdownOpen,
        priorityDropdownOpen,
        currentStatus,
        currentPriority,
        formatDate,
        handleBackToList,
        handleTypeReply,
        handleReviewReply,
        handleAssign,
        handleMerge,
        toggleStatusDropdown,
        togglePriorityDropdown,
        handleSetStatus,
        handleSetPriority
    }
  }
}
</script>