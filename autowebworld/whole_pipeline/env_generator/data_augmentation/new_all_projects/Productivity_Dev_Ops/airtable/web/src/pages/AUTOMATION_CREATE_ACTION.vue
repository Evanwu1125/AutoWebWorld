<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center p-6">
    <div class="bg-white rounded-xl shadow-xl w-full max-w-2xl overflow-hidden flex flex-col h-[600px]">
      
      <!-- Progress Bar -->
      <div class="h-2 bg-gray-100 w-full flex">
         <div class="h-full bg-blue-600 w-full"></div>
      </div>

      <div class="p-8 flex-1 flex flex-col">
         <div class="flex justify-between items-center mb-8">
           <h1 class="text-2xl font-bold text-gray-900">Add an action</h1>
           <button id="back-trigger-step" @click="goBack" class="text-gray-400 hover:text-gray-600 flex items-center gap-1">
             <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
               <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
             </svg>
             Back
           </button>
         </div>

         <div class="flex-1 space-y-6">
           <div class="bg-blue-50 p-4 rounded-lg border border-blue-100 mb-6 flex items-center gap-3">
             <span class="text-blue-500">ℹ️</span>
             <span class="text-sm text-blue-800">Trigger selected: <span class="font-bold">{{ formatTrigger(triggerType) }}</span></span>
           </div>

           <div class="relative">
             <label class="block text-sm font-medium text-gray-700 mb-2">Action</label>
             <button 
               id="action-dropdown"
               @click="dropdownOpen = !dropdownOpen"
               class="w-full flex items-center justify-between px-4 py-3 border border-gray-300 rounded-lg bg-white hover:border-blue-400 transition-colors shadow-sm"
             >
               <span class="flex items-center gap-2">
                  <span v-if="selectedAction" class="text-xl">
                    <span v-if="selectedAction === 'send-email'">📧</span>
                    <span v-else-if="selectedAction === 'update-record'">✏️</span>
                    <span v-else>➕</span>
                  </span>
                  {{ selectedActionLabel || 'Select an action...' }}
               </span>
               <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                 <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
               </svg>
             </button>
             
             <div v-if="dropdownOpen" class="absolute top-full left-0 w-full mt-2 bg-white border border-gray-200 rounded-lg shadow-xl z-20 overflow-hidden">
                <div id="action-send-email" @click="selectAction('send-email')" class="p-4 hover:bg-blue-50 cursor-pointer border-b border-gray-100 flex items-center gap-3">
                   <div class="bg-blue-100 p-2 rounded text-xl">📧</div>
                   <div>
                      <div class="font-bold text-gray-900">Send email</div>
                      <div class="text-xs text-gray-500">Send an email notification</div>
                   </div>
                </div>
                <div id="action-update-record" @click="selectAction('update-record')" class="p-4 hover:bg-blue-50 cursor-pointer border-b border-gray-100 flex items-center gap-3">
                   <div class="bg-green-100 p-2 rounded text-xl">✏️</div>
                   <div>
                      <div class="font-bold text-gray-900">Update record</div>
                      <div class="text-xs text-gray-500">Update a field in a record</div>
                   </div>
                </div>
                <div id="action-create-record" @click="selectAction('create-record')" class="p-4 hover:bg-blue-50 cursor-pointer flex items-center gap-3">
                   <div class="bg-purple-100 p-2 rounded text-xl">➕</div>
                   <div>
                      <div class="font-bold text-gray-900">Create record</div>
                      <div class="text-xs text-gray-500">Create a new record in a table</div>
                   </div>
                </div>
             </div>
           </div>
           
           <!-- Conditional Input -->
           <div v-if="selectedAction === 'send-email'">
              <label class="block text-sm font-medium text-gray-700 mb-2">Recipient Email</label>
              <input 
                id="email-recipient-input"
                v-model="emailRecipient"
                @input="handleEmailInput"
                type="text" 
                class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
                placeholder="recipient@example.com"
              >
           </div>
         </div>

         <div class="flex justify-end pt-6 border-t border-gray-100">
            <button 
               id="save-automation-button"
               @click="saveAutomation"
               class="bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-8 rounded-lg shadow-md transition-all flex items-center gap-2"
               :disabled="!selectedAction"
               :class="{'opacity-50 cursor-not-allowed': !selectedAction}"
            >
               Create Automation
               <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                 <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
               </svg>
            </button>
         </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'AUTOMATION_CREATE_ACTION',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const selectedAction = ref('')
    const dropdownOpen = ref(false)
    const emailRecipient = ref('')
    const triggerType = computed(() => store.trigger_type)

    const selectedActionLabel = computed(() => {
       if (selectedAction.value === 'send-email') return 'Send email'
       if (selectedAction.value === 'update-record') return 'Update record'
       if (selectedAction.value === 'create-record') return 'Create record'
       return ''
    })

    const selectAction = (val) => {
      selectedAction.value = val
      store.action_type = val
      dropdownOpen.value = false
    }

    const handleEmailInput = () => {
       store.email_recipient = emailRecipient.value
    }
    
    const formatTrigger = (t) => {
      if (t === 'when-record-created') return 'When record created'
      if (t === 'when-record-updated') return 'When record updated'
      if (t === 'at-scheduled-time') return 'At scheduled time'
      return t
    }

    const goBack = async () => {
      store.setCurrentPageId('AUTOMATION_CREATE_TRIGGER')
      await router.push({ name: 'AUTOMATION_CREATE_TRIGGER' })
    }

    const saveAutomation = async () => {
      // Create effect
      const newId = 'auto_' + Date.now()
      const newAuto = {
        id: newId,
        base_id: store.selected_base_id,
        name: 'New Automation',
        trigger: store.trigger_type,
        action: store.action_type,
        active: true,
        image: '/images/Automation.jpg'
      }
      
      dataStore.automations.push(newAuto)
      store.created_automation_id = newId
      
      store.setCurrentPageId('AUTOMATION_CREATED_SUCCESS')
      await router.push({ name: 'AUTOMATION_CREATED_SUCCESS' })
    }

    return {
      selectedAction,
      selectedActionLabel,
      dropdownOpen,
      emailRecipient,
      triggerType,
      selectAction,
      handleEmailInput,
      formatTrigger,
      goBack,
      saveAutomation
    }
  }
}
</script>