<template>
  <div class="min-h-screen bg-gray-100 flex items-center justify-center p-4">
    <div class="bg-white rounded-lg shadow-xl w-full max-w-2xl overflow-hidden border-t-8 border-purple-600 p-12 text-center">
      
      <div class="w-20 h-20 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-6">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-10 w-10 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
        </svg>
      </div>

      <h1 class="text-3xl font-bold text-gray-900 mb-4">Thank You!</h1>
      <p class="text-gray-600 mb-10 text-lg">Your response has been successfully recorded.</p>
      
      <div class="flex flex-col sm:flex-row gap-4 justify-center">
        <button 
          id="view-grid-button" 
          @click="goGrid" 
          class="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-6 rounded shadow transition-colors"
        >
          View in Grid
        </button>
        <button 
          id="go-home" 
          @click="goHome" 
          class="bg-white border border-gray-300 hover:bg-gray-50 text-gray-700 font-semibold py-3 px-6 rounded shadow-sm transition-colors"
        >
          Back Home
        </button>
      </div>
      
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'FORM_SUBMISSION_SUCCESS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const goGrid = async () => {
      store.setCurrentPageId('TABLE_GRID_VIEW')
      await router.push({ name: 'TABLE_GRID_VIEW' })
    }
    
    const goHome = async () => {
      store.setCurrentPageId('HOME')
      await router.push({ name: 'HOME' })
    }

    return { goGrid, goHome }
  }
}
</script>