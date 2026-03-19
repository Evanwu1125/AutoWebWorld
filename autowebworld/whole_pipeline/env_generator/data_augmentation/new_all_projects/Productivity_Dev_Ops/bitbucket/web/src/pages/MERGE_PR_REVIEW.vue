<template>
  <div class="min-h-screen bg-[#FAFBFC] flex items-center justify-center py-12 px-4">
    <div class="max-w-md w-full bg-white p-8 rounded-lg shadow-md border border-gray-200">
      <h2 class="text-2xl font-bold text-[#172B4D] mb-6 text-center">Confirm Merge</h2>
      
      <div class="bg-yellow-50 border-l-4 border-yellow-400 p-4 mb-6">
        <p class="text-sm text-yellow-700">
          This action will merge the pull request into the target branch.
        </p>
      </div>

      <div class="space-y-4 mb-8">
        <div>
           <span class="block text-xs font-bold text-gray-500 uppercase">Strategy</span>
           <span class="font-medium capitalize">{{ signatureStore.merge_strategy?.replace('_', ' ') }}</span>
        </div>
        <div>
           <span class="block text-xs font-bold text-gray-500 uppercase">Message</span>
           <span class="font-medium">{{ signatureStore.merge_commit_message }}</span>
        </div>
      </div>

      <div class="flex space-x-4">
        <button 
          id="merge-pr-review-back" 
          @click="goBack"
          class="flex-1 py-2 px-4 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none"
        >
          Back
        </button>
        <button 
          id="merge-pr-confirm" 
          @click="confirm"
          class="flex-1 py-2 px-4 border border-transparent rounded-md text-sm font-medium text-white bg-[#0052CC] hover:bg-blue-700 focus:outline-none shadow-sm"
        >
          Confirm Merge
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { useRouter, useRoute } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'MERGE_PR_REVIEW',
  setup() {
    const router = useRouter()
    const route = useRoute()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const goBack = async () => {
      signatureStore.currentPageId = 'MERGE_PR_FORM'
      await router.push({ name: 'MERGE_PR_FORM', params: route.params })
    }

    const confirm = async () => {
      const prId = route.params.pr_id || signatureStore.selected_pr_id
      const prIndex = dataStore.pull_requests.findIndex(p => p.id === prId)
      
      if (prIndex !== -1) {
        dataStore.pull_requests[prIndex].status = 'merged'
        dataStore.pull_requests[prIndex].updated_at = new Date().toISOString().split('T')[0]
      }

      signatureStore.success_message = `Pull Request merged successfully!`
      signatureStore.currentPageId = 'MERGE_PR_SUCCESS'
      await router.push({ name: 'MERGE_PR_SUCCESS', params: route.params })
    }

    return {
      signatureStore,
      goBack,
      confirm
    }
  }
}
</script>