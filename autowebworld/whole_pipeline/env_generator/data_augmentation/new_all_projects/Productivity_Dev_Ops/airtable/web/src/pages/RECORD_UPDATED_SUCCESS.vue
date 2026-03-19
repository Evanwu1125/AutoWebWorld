<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center p-6">
    <div class="bg-white rounded-xl shadow-lg p-8 text-center max-w-sm w-full">
      <div class="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
        </svg>
      </div>
      <h2 class="text-xl font-bold text-gray-900 mb-6">Record Updated</h2>
      
      <div class="space-y-3">
        <button 
          id="view-record-button" 
          @click="goDetail" 
          class="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors"
        >
          View Record
        </button>
        <button 
          id="go-home" 
          @click="goHome" 
          class="w-full bg-white border border-gray-300 hover:bg-gray-50 text-gray-700 font-semibold py-2 px-4 rounded-lg transition-colors"
        >
          Go Home
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'RECORD_UPDATED_SUCCESS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const goDetail = async () => {
      store.setCurrentPageId('RECORD_DETAIL')
      await router.push({ name: 'RECORD_DETAIL' })
    }
    
    const goHome = async () => {
      store.setCurrentPageId('HOME')
      await router.push({ name: 'HOME' })
    }

    return { goDetail, goHome }
  }
}
</script>