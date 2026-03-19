<template>
  <div class="min-h-screen bg-slate-50 font-inter text-slate-900 flex flex-col">
    <header class="bg-white shadow-sm z-20">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center">
         <h1 class="text-xl font-bold text-slate-900">Review Ticket</h1>
      </div>
    </header>

    <main class="flex-1 max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
       <div class="bg-white p-8 rounded-lg shadow-sm border border-slate-200 space-y-6">
          <div class="border-b border-slate-100 pb-4">
             <h3 class="text-sm font-medium text-slate-500 uppercase tracking-wider mb-1">Subject</h3>
             <p class="text-lg font-medium text-slate-900">{{ signatureStore.new_ticket_subject === 'has_subject' ? 'New Ticket Subject (Simulated)' : 'No Subject' }}</p>
          </div>
          <div class="border-b border-slate-100 pb-4">
             <h3 class="text-sm font-medium text-slate-500 uppercase tracking-wider mb-1">Description</h3>
             <p class="text-base text-slate-800">{{ signatureStore.new_ticket_description === 'has_description' ? 'Detailed description text would go here...' : 'No Description' }}</p>
          </div>
          <div class="grid grid-cols-2 gap-4 border-b border-slate-100 pb-4">
             <div>
                <h3 class="text-sm font-medium text-slate-500 uppercase tracking-wider mb-1">Priority</h3>
                <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                  {{ signatureStore.new_ticket_priority || 'Medium' }}
                </span>
             </div>
             <div>
                <h3 class="text-sm font-medium text-slate-500 uppercase tracking-wider mb-1">Group</h3>
                <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-slate-100 text-slate-800">
                  {{ signatureStore.new_ticket_group || 'Support' }}
                </span>
             </div>
          </div>

          <div class="flex justify-between pt-4">
             <button id="back-new-ticket-form" 
                     @click="handleBack" 
                     class="bg-white border border-slate-300 text-slate-700 hover:bg-slate-50 font-medium py-2 px-4 rounded-md shadow-sm transition-colors">
                Edit
             </button>
             <button id="btn-submit-new-ticket" 
                     @click="handleSubmit" 
                     class="bg-green-600 hover:bg-green-700 text-white font-medium py-2 px-6 rounded-md shadow-sm transition-colors">
                Submit Ticket
             </button>
          </div>
       </div>
    </main>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'NEW_TICKET_REVIEW',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const handleBack = async () => {
        signatureStore.setCurrentPageId('NEW_TICKET_FORM')
        await router.push({ name: 'NEW_TICKET_FORM' })
    }

    const handleSubmit = async () => {
        // Create actual ticket in mock data to persist the change
        const newTicket = {
            id: `t${Date.now()}`,
            subject: 'New Ticket Subject (Simulated)',
            description: 'Detailed description text...',
            status: 'Open',
            priority: signatureStore.new_ticket_priority || 'Medium',
            group: signatureStore.new_ticket_group || 'Support',
            requester_id: 'c1', // Simulated logged in user
            assignee_id: null,
            created_at: new Date().toISOString(),
            image: '/images/SupportTicket.jpg' // Default image
        }
        dataStore.addTicket(newTicket)

        signatureStore.created_ticket_id = newTicket.id
        signatureStore.setCurrentPageId('TICKET_CREATION_SUCCESS')
        await router.push({ name: 'TICKET_CREATION_SUCCESS' })
    }

    return {
        signatureStore,
        handleBack,
        handleSubmit
    }
  }
}
</script>