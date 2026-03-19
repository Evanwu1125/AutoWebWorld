<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center p-4">
    <div class="max-w-md w-full bg-white rounded-xl shadow-lg overflow-hidden">
      <!-- Header -->
      <div class="bg-indigo-600 px-6 py-4 flex items-center justify-between">
        <h1 class="text-xl font-bold text-white">Add Section</h1>
        <button 
            id="section-create-back"
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
        <div>
          <label for="section-name-input" class="block text-sm font-medium text-gray-700 mb-1">Section Name</label>
          <input 
            id="section-name-input"
            v-model="name"
            type="text" 
            class="w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 text-base py-2 px-3 border"
            placeholder="e.g., In Progress"
          >
        </div>

        <div class="pt-2 flex justify-end">
           <button 
             id="section-create-submit"
             @click="submit"
             :disabled="!name"
             class="bg-indigo-600 text-white px-6 py-2 rounded-md font-medium hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 shadow-md transition-all disabled:opacity-50 disabled:cursor-not-allowed"
           >
             Add Section
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

export default {
  name: 'SECTION_CREATE_FORM',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const name = ref('')

    const submit = async () => {
       if (!name.value) return
       
       signatureStore.new_section_name = name.value
       
       const newSection = {
           id: `s${Date.now()}`,
           name: name.value,
           project_id: signatureStore.selected_project_id || 'p1'
       }
       dataStore.addSection(newSection)

       await router.push({ name: 'SECTION_CREATE_SUCCESS' })
    }

    const goBack = async () => {
       await router.push({ name: 'PROJECT_BOARD' })
    }

    return {
       name,
       submit,
       goBack
    }
  }
}
</script>