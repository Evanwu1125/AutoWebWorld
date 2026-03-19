<template>
  <div class="min-h-screen bg-gray-100 flex items-center justify-center p-4">
    <!-- Form Container -->
    <div class="bg-white rounded-lg shadow-xl w-full max-w-2xl overflow-hidden border-t-8 border-purple-600">
      
      <!-- Form Header -->
      <div class="p-8 pb-4">
        <h1 class="text-3xl font-bold text-gray-900 mb-2">Submit Feedback</h1>
        <p class="text-gray-600">Please fill out the form below. Your response will be recorded in our database.</p>
      </div>

      <div class="p-8 space-y-8">
        
        <!-- Name Field -->
        <div>
           <label class="block font-bold text-gray-900 mb-2">Full Name <span class="text-red-500">*</span></label>
           <input 
             id="form-name-input"
             v-model="name"
             @input="handleNameInput"
             type="text" 
             class="w-full px-4 py-3 border border-gray-300 rounded focus:ring-1 focus:ring-purple-500 focus:border-purple-500 outline-none transition-colors"
             placeholder="Your answer"
           >
        </div>

        <!-- Email Field -->
        <div>
           <label class="block font-bold text-gray-900 mb-2">Email Address <span class="text-red-500">*</span></label>
           <input 
             id="form-email-input"
             v-model="email"
             @input="handleEmailInput"
             type="email" 
             class="w-full px-4 py-3 border border-gray-300 rounded focus:ring-1 focus:ring-purple-500 focus:border-purple-500 outline-none transition-colors"
             placeholder="Your answer"
           >
        </div>

        <!-- Status Field (Dropdown) -->
        <div class="relative">
           <label class="block font-bold text-gray-900 mb-2">Status</label>
           <button 
             id="form-status-dropdown"
             @click="dropdownOpen = !dropdownOpen"
             class="w-full flex items-center justify-between px-4 py-3 border border-gray-300 rounded bg-white hover:border-purple-400 text-left"
           >
             <span :class="{'text-gray-500': !selectedStatus}">{{ selectedStatus || 'Select an option' }}</span>
             <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
               <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
             </svg>
           </button>
           
           <div v-if="dropdownOpen" class="absolute top-full left-0 w-full mt-1 bg-white border border-gray-200 rounded shadow-lg z-20">
              <div id="form-status-new" @click="selectStatus('New')" class="p-3 hover:bg-purple-50 cursor-pointer">New</div>
              <div id="form-status-submitted" @click="selectStatus('Submitted')" class="p-3 hover:bg-purple-50 cursor-pointer">Submitted</div>
              <div id="form-status-closed" @click="selectStatus('Closed')" class="p-3 hover:bg-purple-50 cursor-pointer">Closed</div>
           </div>
        </div>

        <!-- Actions -->
        <div class="pt-4 flex items-center justify-between">
           <button 
             id="form-submit-button"
             @click="submitForm"
             class="bg-purple-600 hover:bg-purple-700 text-white font-bold py-3 px-8 rounded shadow transition-colors"
             :disabled="!name || !email"
             :class="{'opacity-50 cursor-not-allowed': !name || !email}"
           >
             Submit
           </button>
           
           <button id="back-record-detail" @click="goBack" class="text-gray-400 hover:text-gray-600 text-sm">
             Back to Airtable
           </button>
        </div>
      </div>
      
      <!-- Footer -->
      <div class="bg-gray-50 p-4 text-center text-xs text-gray-400">
        Powered by Airtable Clone
      </div>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'FORM_VIEW_SUBMISSION',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const name = ref('')
    const email = ref('')
    const selectedStatus = ref('')
    const dropdownOpen = ref(false)

    const handleNameInput = () => {
      store.form_name = name.value
    }

    const handleEmailInput = () => {
      store.form_email = email.value
    }

    const selectStatus = (val) => {
      selectedStatus.value = val
      store.form_status = val
      dropdownOpen.value = false
    }

    const submitForm = async () => {
      // Mock submission - creates a record
      const newId = 'sub_' + Date.now()
      // We might need a table ID context, assume one exists or pick first
      const tableId = store.selected_table_id || (dataStore.tables[0] ? dataStore.tables[0].id : null)
      
      if (tableId) {
        dataStore.records.push({
          id: newId,
          table_id: tableId,
          title: store.form_name,
          status: store.form_status || 'Submitted',
          due_date: new Date().toISOString().split('T')[0],
          priority: 'Medium',
          assigned_to: store.form_email, // Map email to assignee for demo
          image: '/images/FormSubmission.jpg'
        })
      }
      
      store.submitted_record_id = newId
      
      store.setCurrentPageId('FORM_SUBMISSION_SUCCESS')
      await router.push({ name: 'FORM_SUBMISSION_SUCCESS' })
    }
    
    const goBack = async () => {
      // Per FSM: ACT_FORM_BACK_TO_DETAIL -> RECORD_DETAIL
      store.setCurrentPageId('RECORD_DETAIL')
      await router.push({ name: 'RECORD_DETAIL' })
    }

    return {
      name,
      email,
      selectedStatus,
      dropdownOpen,
      handleNameInput,
      handleEmailInput,
      selectStatus,
      submitForm,
      goBack
    }
  }
}
</script>