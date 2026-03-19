<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center p-4">
    <div class="max-w-md w-full bg-white rounded-xl shadow-lg p-8 text-center">
      <div class="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-6">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-10 w-10 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
        </svg>
      </div>
      
      <h2 class="text-2xl font-bold text-gray-900 mb-2">Success!</h2>
      <p class="text-gray-600 mb-8">
        Your project has been created successfully. What would you like to do next?
      </p>
      
      <div class="space-y-3">
        <button 
          id="success-back-projects"
          @click="goProjects"
          class="w-full bg-indigo-600 text-white px-4 py-3 rounded-lg font-medium hover:bg-indigo-700 transition-colors shadow-sm"
        >
          View All Projects
        </button>
        
        <button 
          id="success-go-home"
          @click="goHome"
          class="w-full bg-white text-gray-700 border border-gray-300 px-4 py-3 rounded-lg font-medium hover:bg-gray-50 transition-colors"
        >
          Go Home
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'PROJECT_CREATE_SUCCESS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    onMounted(() => {
        // Effect: set success_message
        store.success_message = "Project created successfully"
    })

    const goProjects = async () => {
      await router.push({ name: 'PROJECTS_LIST' })
    }

    const goHome = async () => {
      await router.push({ name: 'HOME' })
    }

    return {
      goProjects,
      goHome
    }
  }
}
</script>