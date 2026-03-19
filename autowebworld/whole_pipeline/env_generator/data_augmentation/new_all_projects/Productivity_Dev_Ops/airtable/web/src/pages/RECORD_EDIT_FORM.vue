<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center p-6">
    <div class="bg-white rounded-xl shadow-xl w-full max-w-lg p-8">
      <div class="flex justify-between items-center mb-8">
        <h1 class="text-2xl font-bold text-gray-900">Edit Record</h1>
        <button id="back-record-detail" @click="goBack" class="text-gray-400 hover:text-gray-600">
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
            id="edit-title-input"
            v-model="title"
            @input="handleTitleInput"
            type="text" 
            class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
          >
        </div>

        <!-- Status Dropdown -->
        <div class="relative">
          <label class="block text-sm font-medium text-gray-700 mb-2">Status</label>
          <button 
            id="edit-status-dropdown"
            @click="statusDropdownOpen = !statusDropdownOpen"
            class="w-full flex items-center justify-between px-4 py-2 border border-gray-300 rounded-lg bg-white hover:border-blue-400"
          >
            <span>{{ selectedStatus || 'Select Status' }}</span>
            <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          
          <div v-if="statusDropdownOpen" class="absolute top-full left-0 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-xl z-20">
             <div id="edit-status-to-do" @click="selectStatus('To do')" class="p-3 hover:bg-gray-50 cursor-pointer">To do</div>
             <div id="edit-status-in-progress" @click="selectStatus('In progress')" class="p-3 hover:bg-gray-50 cursor-pointer">In progress</div>
             <div id="edit-status-done" @click="selectStatus('Done')" class="p-3 hover:bg-gray-50 cursor-pointer">Done</div>
          </div>
        </div>

        <!-- Submit -->
        <button 
          id="save-record-button"
          @click="saveRecord"
          class="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-lg shadow-md transition-colors mt-4"
          :disabled="!title"
          :class="{'opacity-50 cursor-not-allowed': !title}"
        >
          Save Changes
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'RECORD_EDIT_FORM',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const title = ref('')
    const selectedStatus = ref('')
    const statusDropdownOpen = ref(false)

    onMounted(() => {
      const record = dataStore.records.find(r => r.id === store.selected_record_id)
      if (record) {
        title.value = record.title
        selectedStatus.value = record.status
        store.edit_title_input = record.title
        store.edit_status_select = record.status
      }
    })

    const handleTitleInput = () => {
      store.edit_title_input = title.value
    }

    const selectStatus = (status) => {
      selectedStatus.value = status
      store.edit_status_select = status
      statusDropdownOpen.value = false
    }

    const saveRecord = async () => {
      // Mock update
      const index = dataStore.records.findIndex(r => r.id === store.selected_record_id)
      if (index !== -1) {
        dataStore.records[index] = {
           ...dataStore.records[index],
           title: store.edit_title_input,
           status: store.edit_status_select
        }
      }
      store.updated_record_id = store.selected_record_id
      
      store.setCurrentPageId('RECORD_UPDATED_SUCCESS')
      await router.push({ name: 'RECORD_UPDATED_SUCCESS' })
    }

    const goBack = async () => {
      store.setCurrentPageId('RECORD_DETAIL')
      await router.push({ name: 'RECORD_DETAIL' })
    }

    return {
      title,
      selectedStatus,
      statusDropdownOpen,
      handleTitleInput,
      selectStatus,
      saveRecord,
      goBack
    }
  }
}
</script>