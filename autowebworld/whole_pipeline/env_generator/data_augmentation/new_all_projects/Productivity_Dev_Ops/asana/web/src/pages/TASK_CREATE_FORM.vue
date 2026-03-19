<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center p-4">
    <div class="max-w-xl w-full bg-white rounded-xl shadow-lg overflow-hidden">
      <!-- Header -->
      <div class="bg-indigo-600 px-6 py-4 flex items-center justify-between">
        <h1 class="text-xl font-bold text-white">New Task</h1>
        <button 
            id="task-create-back"
            @click="goBack"
            class="text-indigo-200 hover:text-white transition-colors"
        >
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
        </button>
      </div>

      <!-- Form -->
      <div class="p-8 space-y-6">
        <!-- Name -->
        <div>
          <label for="task-name-input" class="block text-sm font-medium text-gray-700 mb-1">Task Name</label>
          <input 
            id="task-name-input"
            v-model="name"
            type="text" 
            class="w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 text-base py-2 px-3 border"
            placeholder="e.g., Draft Q4 Report"
          >
        </div>

        <!-- Description -->
        <div>
          <label for="task-description-input" class="block text-sm font-medium text-gray-700 mb-1">Description</label>
          <textarea 
            id="task-description-input"
            v-model="description"
            rows="3"
            class="w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 text-base py-2 px-3 border"
            placeholder="Details about this task..."
          ></textarea>
        </div>

        <!-- Due Date -->
        <div>
           <label class="block text-sm font-medium text-gray-700 mb-1">Due Date</label>
           <DateTimePicker 
             id="date-picker1" 
             @change="handleDateChange" 
           />
        </div>

        <!-- Assignee (Hover Menu Widget) -->
        <div class="relative">
           <label class="block text-sm font-medium text-gray-700 mb-1">Assignee</label>
           <button 
             id="assignee-dropdown"
             @click="toggleAssigneeMenu"
             class="w-full text-left rounded-md border border-gray-300 shadow-sm px-3 py-2 bg-white text-sm font-medium text-gray-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 flex justify-between items-center"
           >
             {{ assigneeLabel }}
             <svg class="h-5 w-5 text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
               <path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" />
             </svg>
           </button>

           <div v-if="assigneeMenuOpen" class="absolute z-10 mt-1 w-full bg-white shadow-lg max-h-56 rounded-md py-1 text-base ring-1 ring-black ring-opacity-5 overflow-auto focus:outline-none sm:text-sm">
              <div 
                id="assignee-option-me"
                @click="selectAssignee('me')"
                class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-indigo-50 text-gray-900"
              >
                <div class="flex items-center">
                   <img src="/images/photo1765161065.jpg" alt="" class="flex-shrink-0 h-6 w-6 rounded-full" />
                   <span class="ml-3 block truncate">Me</span>
                </div>
              </div>
              <div 
                id="assignee-option-unassigned"
                @click="selectAssignee('unassigned')"
                class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-indigo-50 text-gray-900"
              >
                 <span class="ml-3 block truncate">Unassigned</span>
              </div>
           </div>
        </div>

        <!-- Submit -->
        <div class="pt-4 flex justify-end">
           <button 
             id="task-create-submit"
             @click="submit"
             :disabled="!name"
             class="bg-indigo-600 text-white px-6 py-2 rounded-md font-medium hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 shadow-md transition-all disabled:opacity-50 disabled:cursor-not-allowed"
           >
             Create Task
           </button>
        </div>
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
  name: 'TASK_CREATE_FORM',
  components: {
    DateTimePicker
  },
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const name = ref('')
    const description = ref('')
    const dueDate = ref(null)
    const assignee = ref('unassigned')
    const assigneeMenuOpen = ref(false)

    const assigneeLabel = ref('Unassigned')

    const handleDateChange = (date) => {
       dueDate.value = date
       signatureStore.new_task_due_date = date
    }

    const toggleAssigneeMenu = () => {
       assigneeMenuOpen.value = !assigneeMenuOpen.value
    }

    const selectAssignee = (val) => {
       assignee.value = val
       assigneeLabel.value = val === 'me' ? 'Me' : 'Unassigned'
       signatureStore.new_task_assignee = val === 'me' ? 'me' : null
       assigneeMenuOpen.value = false
    }

    const submit = async () => {
       if (!name.value) return
       
       signatureStore.new_task_name = name.value
       signatureStore.new_task_description = description.value

       const newTask = {
          id: `t${Date.now()}`,
          name: name.value,
          description: description.value,
          due_date: dueDate.value,
          assignee_id: assignee.value === 'me' ? 'u1' : null,
          project_id: signatureStore.selected_project_id || 'p1', // Default to current or first
          section_id: 's1', // Default to first section of project (simplified)
          priority: 50,
          completed: false,
          image: '/images/Task.jpg'
       }
       dataStore.addTask(newTask)

       await router.push({ name: 'TASK_CREATE_SUCCESS' })
    }

    const goBack = async () => {
       await router.push({ name: 'PROJECT_BOARD' })
    }

    return {
       name,
       description,
       assigneeLabel,
       assigneeMenuOpen,
       handleDateChange,
       toggleAssigneeMenu,
       selectAssignee,
       submit,
       goBack
    }
  }
}
</script>