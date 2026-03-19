<template>
  <div class="min-h-screen bg-[#FAFBFC] flex items-center justify-center py-12 px-4">
    <div class="max-w-md w-full bg-white p-8 rounded-lg shadow-md border border-gray-200">
      <h2 class="text-2xl font-bold text-[#172B4D] mb-6 text-center">Review Pull Request</h2>

      <div class="space-y-4 mb-8 bg-gray-50 p-6 rounded-md border border-gray-100">
         <div>
           <span class="block text-xs font-bold text-gray-500 uppercase">Title</span>
           <span class="text-gray-900 font-medium">{{ signatureStore.pr_title }}</span>
         </div>
         <div class="flex items-center space-x-2">
            <div>
               <span class="block text-xs font-bold text-gray-500 uppercase">Source</span>
               <span class="bg-gray-200 px-2 py-1 rounded text-sm">{{ signatureStore.pr_source_branch }}</span>
            </div>
            <div class="text-gray-400">→</div>
            <div>
               <span class="block text-xs font-bold text-gray-500 uppercase">Target</span>
               <span class="bg-gray-200 px-2 py-1 rounded text-sm">{{ signatureStore.pr_target_branch }}</span>
            </div>
         </div>
      </div>

      <div class="flex space-x-4">
        <button 
          id="create-pr-review-back" 
          @click="goBack"
          class="flex-1 py-2 px-4 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none"
        >
          Back
        </button>
        <button 
          id="create-pr-confirm" 
          @click="confirm"
          class="flex-1 py-2 px-4 border border-transparent rounded-md text-sm font-medium text-white bg-[#0052CC] hover:bg-blue-700 focus:outline-none shadow-sm"
        >
          Create
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
  name: 'CREATE_PR_REVIEW',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const goBack = async () => {
      signatureStore.currentPageId = 'CREATE_PR_FORM'
      await router.push({ name: 'CREATE_PR_FORM' })
    }

    const confirm = async () => {
      const newPR = {
        id: `pr_${Date.now()}`,
        title: signatureStore.pr_title,
        author_id: 'user_001',
        repo_id: 'repo_001', // Default for now
        status: 'open',
        created_at: new Date().toISOString().split('T')[0],
        updated_at: new Date().toISOString().split('T')[0],
        image: '/images/photo1765608733.jpg'
      }
      
      dataStore.pull_requests.unshift(newPR)
      signatureStore.success_message = `Pull Request "${newPR.title}" created successfully!`
      
      signatureStore.currentPageId = 'CREATE_PR_SUCCESS'
      await router.push({ name: 'CREATE_PR_SUCCESS' })
    }

    return {
      signatureStore,
      goBack,
      confirm
    }
  }
}
</script>