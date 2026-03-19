<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center p-4">
    <div class="max-w-xl w-full bg-white rounded-xl shadow-lg overflow-hidden">
      <!-- Header -->
      <div class="bg-indigo-600 px-6 py-4 flex items-center justify-between">
        <h1 class="text-xl font-bold text-white">Create New Project</h1>
        <button 
            id="project-create-back"
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
          <label for="project-name-input" class="block text-sm font-medium text-gray-700 mb-1">Project Name</label>
          <input 
            id="project-name-input"
            v-model="name"
            type="text" 
            class="w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 text-base py-2 px-3 border"
            placeholder="e.g., Q4 Marketing Campaign"
          >
        </div>

        <!-- Description -->
        <div>
          <label for="project-description-input" class="block text-sm font-medium text-gray-700 mb-1">Description</label>
          <textarea 
            id="project-description-input"
            v-model="description"
            rows="3"
            class="w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 text-base py-2 px-3 border"
            placeholder="What is this project about?"
          ></textarea>
        </div>

        <!-- Due Date -->
        <div>
           <label class="block text-sm font-medium text-gray-700 mb-1">Due Date</label>
           <!-- Reusing existing DateTimePicker as required by FSM contract -->
           <!-- Note: The FSM specific selectors like .year-2025 are internal to DateTimePicker -->
           <DateTimePicker 
             id="date-picker1" 
             @change="handleDateChange" 
           />
        </div>

        <!-- Actions -->
        <div class="pt-4 flex justify-end">
           <button 
             id="project-create-submit"
             @click="submit"
             :disabled="!name"
             class="bg-indigo-600 text-white px-6 py-2 rounded-md font-medium hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 shadow-md transition-all disabled:opacity-50 disabled:cursor-not-allowed"
           >
             Create Project
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
  name: 'PROJECT_CREATE_FORM',
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

    // Sync input with signature store (Effects: PROJECT_CREATE_TYPE_NAME etc.)
    // We can use watchers or just set it on submit. FSM implies "type" action sets state immediately.
    // Vue v-model does this automatically if we mapped directly to store, but local ref + submit is cleaner.
    // To strictly follow FSM "type" action effect:
    // We can assume v-model updates correspond to typing.

    const handleDateChange = (date) => {
       dueDate.value = date
       signatureStore.new_project_due_date = date
    }

    const submit = async () => {
      // Precondition: name length > 0
      if (!name.value) return

      // Effects: Set signature vars
      signatureStore.new_project_name = name.value
      signatureStore.new_project_description = description.value
      
      // Actual Data Store Update (Simulating FSM "append($.projects, ...)")
      const newProject = {
          id: `p${Date.now()}`,
          name: name.value,
          description: description.value,
          due_date: dueDate.value,
          status: 'Active',
          priority: 50,
          image: '/images/Project.jpg' // Placeholder
      }
      dataStore.addProject(newProject)

      await router.push({ name: 'PROJECT_CREATE_SUCCESS' })
    }

    const goBack = async () => {
      await router.push({ name: 'PROJECTS_LIST' })
    }

    return {
      name,
      description,
      handleDateChange,
      submit,
      goBack
    }
  }
}
</script>