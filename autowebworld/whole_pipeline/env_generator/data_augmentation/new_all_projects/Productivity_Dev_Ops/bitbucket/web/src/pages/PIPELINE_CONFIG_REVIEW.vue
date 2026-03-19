<template>
  <div class="min-h-screen bg-[#FAFBFC] flex items-center justify-center py-12 px-4">
    <div class="max-w-md w-full bg-white p-8 rounded-lg shadow-md border border-gray-200">
      <h2 class="text-2xl font-bold text-[#172B4D] mb-6 text-center">Review Pipeline Configuration</h2>
      
      <div class="space-y-4 mb-8 bg-gray-50 p-6 rounded-md border border-gray-100">
         <div class="flex justify-between border-b border-gray-200 pb-2">
           <span class="text-gray-500 font-medium">Name</span>
           <span class="text-[#172B4D] font-bold">{{ signatureStore.pipeline_name }}</span>
         </div>
         <div class="flex justify-between border-b border-gray-200 pb-2">
           <span class="text-gray-500 font-medium">Branch</span>
           <span class="bg-gray-200 px-2 py-0.5 rounded text-sm font-mono">{{ signatureStore.pipeline_branch }}</span>
         </div>
         <div class="flex justify-between">
           <span class="text-gray-500 font-medium">Trigger</span>
           <span class="text-[#172B4D]">{{ signatureStore.pipeline_trigger === 'on_push' ? 'On Push' : 'Manual' }}</span>
         </div>
      </div>

      <div class="flex space-x-4">
        <button 
          id="pipeline-config-review-back" 
          @click="goBack"
          class="flex-1 py-2 px-4 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none"
        >
          Back
        </button>
        <button 
          id="pipeline-config-confirm" 
          @click="confirm"
          class="flex-1 py-2 px-4 border border-transparent rounded-md text-sm font-medium text-white bg-[#0052CC] hover:bg-blue-700 focus:outline-none shadow-sm"
        >
          Create Pipeline
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'PIPELINE_CONFIG_REVIEW',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const goBack = async () => {
      signatureStore.currentPageId = 'PIPELINE_CONFIG_FORM'
      await router.push({ name: 'PIPELINE_CONFIG_FORM' })
    }

    const confirm = async () => {
      const newPipe = {
        id: `pipe_${Date.now()}`,
        name: signatureStore.pipeline_name,
        repo_id: 'repo_001', // Default
        status: 'success', // Fake success
        branch: signatureStore.pipeline_branch,
        trigger: signatureStore.pipeline_trigger,
        created_at: new Date().toISOString().replace('T', ' ').substring(0, 16),
        image: '/images/photo1765608818.jpg'
      }
      
      dataStore.pipelines.unshift(newPipe)
      
      signatureStore.success_message = `Pipeline "${newPipe.name}" created successfully!`
      signatureStore.currentPageId = 'CREATE_PIPELINE_SUCCESS'
      await router.push({ name: 'CREATE_PIPELINE_SUCCESS' })
    }

    return {
      signatureStore,
      goBack,
      confirm
    }
  }
}
</script>