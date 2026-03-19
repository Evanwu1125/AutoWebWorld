<template>
  <div class="min-h-screen bg-gray-50 flex flex-col justify-center py-12 sm:px-6 lg:px-8">
    <div class="sm:mx-auto sm:w-full sm:max-w-md">
      <div class="bg-white py-8 px-4 shadow sm:rounded-lg sm:px-10 text-center">
        <div class="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-blue-100 mb-6">
          <svg class="h-6 w-6 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
        </div>
        
        <h2 class="text-2xl font-bold text-gray-900 mb-2">Experiment Scheduled</h2>
        <p class="text-gray-600 mb-8">Your experiment has been scheduled to run automatically.</p>
        
        <div class="space-y-4">
          <button 
            id="btn-view-experiments"
            @click="goToExperiments"
            class="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none"
          >
            View All Experiments
          </button>
          
          <button 
            id="go-home"
            @click="goHome"
            class="w-full flex justify-center py-2 px-4 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none"
          >
            Back to Home
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'EXPERIMENT_SCHEDULED_SUCCESS',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    // FSM Effect simulation
    signatureStore.success_message = "Experiment scheduled successfully"

    function goToExperiments() {
      signatureStore.setCurrentPageId('EXPERIMENTS_LIST')
      router.push({ name: 'EXPERIMENTS_LIST' })
    }

    function goHome() {
      signatureStore.setCurrentPageId('HOME')
      router.push({ name: 'HOME' })
    }

    return {
      goToExperiments,
      goHome
    }
  }
}
</script>