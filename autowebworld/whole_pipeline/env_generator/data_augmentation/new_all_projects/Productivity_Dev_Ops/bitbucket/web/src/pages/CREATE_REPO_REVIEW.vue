<template>
  <div class="min-h-screen bg-[#FAFBFC] flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
    <div class="max-w-md w-full bg-white p-8 rounded-lg shadow-md border border-gray-200">
      <div class="text-center mb-8">
        <h2 class="text-2xl font-bold text-[#172B4D]">Review Repository Details</h2>
        <p class="text-gray-600 mt-2">Please verify the information before creating.</p>
      </div>

      <div class="space-y-4 mb-8 bg-gray-50 p-6 rounded-md border border-gray-100">
        <div class="flex justify-between border-b border-gray-200 pb-2">
          <span class="text-gray-500 font-medium">Name</span>
          <span class="text-[#172B4D] font-bold">{{ signatureStore.repo_name }}</span>
        </div>
        <div class="flex justify-between border-b border-gray-200 pb-2">
          <span class="text-gray-500 font-medium">Access Level</span>
          <span 
            class="px-2 py-0.5 rounded text-xs font-bold uppercase"
            :class="signatureStore.repo_access_level === 'private' ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'"
          >
            {{ signatureStore.repo_access_level }}
          </span>
        </div>
        <div class="flex justify-between items-start">
          <span class="text-gray-500 font-medium">Description</span>
          <span class="text-[#172B4D] text-right max-w-[200px]">{{ signatureStore.repo_description || 'No description' }}</span>
        </div>
      </div>

      <div class="flex space-x-4">
        <button 
          id="create-repo-review-back" 
          @click="goBack"
          class="flex-1 py-2 px-4 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none"
        >
          Back
        </button>
        <button 
          id="create-repo-confirm" 
          @click="confirm"
          class="flex-1 py-2 px-4 border border-transparent rounded-md text-sm font-medium text-white bg-[#0052CC] hover:bg-blue-700 focus:outline-none shadow-sm"
        >
          Confirm & Create
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
  name: 'CREATE_REPO_REVIEW',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const goBack = async () => {
      signatureStore.currentPageId = 'CREATE_REPO_FORM'
      await router.push({ name: 'CREATE_REPO_FORM' })
    }

    const confirm = async () => {
      // Simulate creation logic - add to mock data
      const newRepo = {
        id: `repo_${Date.now()}`,
        name: signatureStore.repo_name,
        owner: 'Team Alpha', // Default owner
        access: signatureStore.repo_access_level,
        description: signatureStore.repo_description,
        updated_at: new Date().toISOString().split('T')[0],
        activity: 0,
        image: '/images/Repository.jpg' // Placeholder
      }
      
      dataStore.repositories.unshift(newRepo)
      
      signatureStore.success_message = `Repository "${newRepo.name}" created successfully!`
      signatureStore.currentPageId = 'CREATE_REPO_SUCCESS'
      await router.push({ name: 'CREATE_REPO_SUCCESS' })
    }

    return {
      signatureStore,
      goBack,
      confirm
    }
  }
}
</script>