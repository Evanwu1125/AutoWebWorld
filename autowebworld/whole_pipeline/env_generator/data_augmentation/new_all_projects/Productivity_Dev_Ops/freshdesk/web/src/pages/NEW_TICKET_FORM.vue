<template>
  <div class="min-h-screen bg-slate-50 font-inter text-slate-900 flex flex-col">
    <header class="bg-white shadow-sm z-20">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center">
         <button id="back-tickets" @click="handleBack" class="mr-4 text-slate-500 hover:text-blue-600 transition-colors">
            ← Back
         </button>
         <h1 class="text-xl font-bold text-slate-900">Create New Ticket</h1>
      </div>
    </header>

    <main class="flex-1 max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
       <div class="bg-white p-8 rounded-lg shadow-sm border border-slate-200 space-y-6">
          
          <!-- Subject -->
          <div>
             <label for="new-ticket-subject" class="block text-sm font-medium text-slate-700 mb-1">Subject</label>
             <input type="text" 
                    id="new-ticket-subject" 
                    v-model="subject" 
                    @input="handleSubjectInput"
                    class="block w-full border-slate-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm p-2.5" 
                    placeholder="Brief summary of the issue">
          </div>

          <!-- Description -->
          <div>
             <label for="new-ticket-description" class="block text-sm font-medium text-slate-700 mb-1">Description</label>
             <textarea id="new-ticket-description" 
                       v-model="description" 
                       @input="handleDescriptionInput"
                       rows="6"
                       class="block w-full border-slate-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm p-2.5 resize-none" 
                       placeholder="Detailed explanation..."></textarea>
          </div>

          <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
             <!-- Priority -->
             <div class="relative">
                <label class="block text-sm font-medium text-slate-700 mb-1">Priority</label>
                <div class="relative">
                   <button id="new-ticket-priority-dropdown" @click="togglePriorityDropdown" class="w-full bg-white border border-slate-300 rounded-md py-2 px-3 text-left shadow-sm focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 sm:text-sm">
                      <span class="block truncate">{{ priority || 'Select Priority' }}</span>
                      <span class="absolute inset-y-0 right-0 flex items-center pr-2 pointer-events-none text-slate-400">▼</span>
                   </button>
                   <div v-if="priorityDropdownOpen" class="absolute z-10 mt-1 w-full bg-white shadow-lg max-h-60 rounded-md py-1 text-base ring-1 ring-black ring-opacity-5 overflow-auto focus:outline-none sm:text-sm">
                      <div id="new-ticket-priority-low" @click="handleSelectPriority('Low')" class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-blue-50 text-slate-900">Low</div>
                      <div id="new-ticket-priority-medium" @click="handleSelectPriority('Medium')" class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-blue-50 text-slate-900">Medium</div>
                      <div id="new-ticket-priority-high" @click="handleSelectPriority('High')" class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-blue-50 text-slate-900">High</div>
                   </div>
                </div>
             </div>

             <!-- Group -->
             <div class="relative">
                <label class="block text-sm font-medium text-slate-700 mb-1">Group</label>
                <div class="relative">
                   <button id="new-ticket-group-dropdown" @click="toggleGroupDropdown" class="w-full bg-white border border-slate-300 rounded-md py-2 px-3 text-left shadow-sm focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 sm:text-sm">
                      <span class="block truncate">{{ group || 'Select Group' }}</span>
                      <span class="absolute inset-y-0 right-0 flex items-center pr-2 pointer-events-none text-slate-400">▼</span>
                   </button>
                   <div v-if="groupDropdownOpen" class="absolute z-10 mt-1 w-full bg-white shadow-lg max-h-60 rounded-md py-1 text-base ring-1 ring-black ring-opacity-5 overflow-auto focus:outline-none sm:text-sm">
                      <div id="new-ticket-group-support" @click="handleSelectGroup('Support')" class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-blue-50 text-slate-900">Support</div>
                      <div id="new-ticket-group-sales" @click="handleSelectGroup('Sales')" class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-blue-50 text-slate-900">Sales</div>
                      <div id="new-ticket-group-billing" @click="handleSelectGroup('Billing')" class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-blue-50 text-slate-900">Billing</div>
                   </div>
                </div>
             </div>
          </div>

          <div class="pt-4 flex justify-end">
             <button id="btn-new-ticket-review" 
                     @click="handleReview"
                     :disabled="!isValid"
                     :class="[
                        'px-6 py-2.5 rounded-md font-medium text-sm transition-colors shadow-sm',
                        isValid ? 'bg-blue-600 text-white hover:bg-blue-700' : 'bg-slate-200 text-slate-400 cursor-not-allowed'
                     ]">
               Review Ticket
             </button>
          </div>
       </div>
    </main>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'NEW_TICKET_FORM',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const subject = ref('')
    const description = ref('')
    const priority = ref('Medium') // Default
    const group = ref('Support')   // Default
    
    const priorityDropdownOpen = ref(false)
    const groupDropdownOpen = ref(false)

    const isValid = computed(() => {
        return subject.value.length > 0 && description.value.length > 0
    })

    const handleSubjectInput = () => {
        if (subject.value) signatureStore.new_ticket_subject = 'has_subject'
        else signatureStore.new_ticket_subject = null
    }

    const handleDescriptionInput = () => {
        if (description.value) signatureStore.new_ticket_description = 'has_description'
        else signatureStore.new_ticket_description = null
    }

    const togglePriorityDropdown = () => priorityDropdownOpen.value = !priorityDropdownOpen.value
    const toggleGroupDropdown = () => groupDropdownOpen.value = !groupDropdownOpen.value

    const handleSelectPriority = (val) => {
        priority.value = val
        signatureStore.new_ticket_priority = val
        priorityDropdownOpen.value = false
    }

    const handleSelectGroup = (val) => {
        group.value = val
        signatureStore.new_ticket_group = val
        groupDropdownOpen.value = false
    }

    const handleReview = async () => {
        if (!isValid.value) return
        signatureStore.setCurrentPageId('NEW_TICKET_REVIEW')
        await router.push({ name: 'NEW_TICKET_REVIEW' })
    }

    const handleBack = async () => {
        signatureStore.setCurrentPageId('TICKETS_LIST')
        await router.push({ name: 'TICKETS_LIST' })
    }

    return {
        subject,
        description,
        priority,
        group,
        priorityDropdownOpen,
        groupDropdownOpen,
        isValid,
        handleSubjectInput,
        handleDescriptionInput,
        togglePriorityDropdown,
        toggleGroupDropdown,
        handleSelectPriority,
        handleSelectGroup,
        handleReview,
        handleBack
    }
  }
}
</script>