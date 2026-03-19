<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center p-6">
    <div class="bg-white rounded-xl shadow-xl w-full max-w-lg p-8">
      <div class="flex justify-between items-center mb-8">
        <h1 class="text-2xl font-bold text-gray-900">New Record</h1>
        <button id="back-grid-view" @click="goBack" class="text-gray-400 hover:text-gray-600">
           <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
             <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
           </svg>
        </button>
      </div>

      <div class="space-y-6">
        <!-- Title Input -->
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">Title</label>
          <input 
            id="field-title-input"
            v-model="title"
            @input="handleTitleInput"
            type="text" 
            class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
            placeholder="Record title"
          >
        </div>

        <!-- Status Dropdown -->
        <div class="relative">
          <label class="block text-sm font-medium text-gray-700 mb-2">Status</label>
          <button 
            id="field-status-dropdown"
            @click="statusDropdownOpen = !statusDropdownOpen"
            class="w-full flex items-center justify-between px-4 py-2 border border-gray-300 rounded-lg bg-white hover:border-blue-400"
          >
            <span>{{ selectedStatus || 'Select Status' }}</span>
            <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          
          <div v-if="statusDropdownOpen" class="absolute top-full left-0 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-xl z-20">
             <div id="status-to-do" @click="selectStatus('To do')" class="p-3 hover:bg-gray-50 cursor-pointer">To do</div>
             <div id="status-in-progress" @click="selectStatus('In progress')" class="p-3 hover:bg-gray-50 cursor-pointer">In progress</div>
             <div id="status-done" @click="selectStatus('Done')" class="p-3 hover:bg-gray-50 cursor-pointer">Done</div>
          </div>
        </div>

        <!-- Date Picker -->
        <div>
           <label class="block text-sm font-medium text-gray-700 mb-2">Due Date</label>
           <!-- Reusing template DateTimePicker -->
           <DateTimePicker id="date-picker" @change="handleDateChange" />
        </div>

        <!-- Submit -->
        <button 
          id="submit-record-button"
          @click="submitRecord"
          class="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-lg shadow-md transition-colors mt-4"
          :disabled="!title || !selectedStatus"
          :class="{'opacity-50 cursor-not-allowed': !title || !selectedStatus}"
        >
          Create Record
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'
import DateTimePicker from '../components/widgets/DateTimePicker.vue'

export default {
  name: 'RECORD_CREATE_FORM',
  components: {
    DateTimePicker
  },
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const title = ref('')
    const selectedStatus = ref('')
    const statusDropdownOpen = ref(false)
    const dueDate = ref('')

    const handleTitleInput = () => {
      store.field_title_input = title.value
    }

    const selectStatus = (status) => {
      selectedStatus.value = status
      store.field_status_select = status
      statusDropdownOpen.value = false
    }

    const handleDateChange = (val) => {
      // Assuming DateTimePicker returns a date object or string
      // FSM expects effect: set field_due_date = now() on click components, but real usage needs value
      // Here we just set the store field
      dueDate.value = val
      store.field_due_date = val
    }

    const submitRecord = async () => {
      // Mock creation effect
      const newId = 'rec_' + Date.now()
      const newRecord = {
        id: newId,
        table_id: store.selected_table_id,
        title: store.field_title_input,
        status: store.field_status_select,
        due_date: store.field_due_date || new Date().toISOString().split('T')[0],
        priority: 'Medium',
        assigned_to: 'Me',
        image: '/images/Record.jpg'
      }
      
      dataStore.records.push(newRecord)
      store.created_record_id = newId
      
      store.setCurrentPageId('RECORD_CREATED_SUCCESS')
      await router.push({ name: 'RECORD_CREATED_SUCCESS' })
    }

    const goBack = async () => {
      store.setCurrentPageId('TABLE_GRID_VIEW')
      await router.push({ name: 'TABLE_GRID_VIEW' })
    }

    return {
      title,
      selectedStatus,
      statusDropdownOpen,
      handleTitleInput,
      selectStatus,
      handleDateChange,
      submitRecord,
      goBack
    }
  }
}
</script>