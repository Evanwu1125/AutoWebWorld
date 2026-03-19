<template>
  <div class="min-h-screen bg-[#FAFBFC] flex items-center justify-center py-12 px-4">
    <div class="max-w-md w-full bg-white p-8 rounded-lg shadow-lg border-t-4 border-green-500 text-center">
      <div class="w-20 h-20 bg-green-100 text-green-500 rounded-full flex items-center justify-center mx-auto mb-6">
        <svg class="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>
      </div>
      
      <h2 class="text-2xl font-bold text-[#172B4D] mb-2">Pipeline Created!</h2>
      <p class="text-gray-600 mb-8">{{ signatureStore.success_message }}</p>

      <button 
        id="pipeline-success-go-home" 
        @click="goHome"
        class="w-full py-3 px-4 border border-transparent rounded-md text-sm font-bold text-white bg-green-600 hover:bg-green-700 focus:outline-none shadow-md transition-colors"
      >
        Go back to Home
      </button>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'CREATE_PIPELINE_SUCCESS',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const goHome = async () => {
      signatureStore.currentPageId = 'HOME'
      await router.push({ name: 'HOME' })
    }

    return {
      signatureStore,
      goHome
    }
  }
}
</script>