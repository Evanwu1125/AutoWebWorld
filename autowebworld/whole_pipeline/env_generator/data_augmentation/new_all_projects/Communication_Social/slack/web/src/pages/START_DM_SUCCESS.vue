<template>
  <div class="h-screen flex flex-col items-center justify-center bg-purple-50">
    <div class="bg-white p-8 rounded-xl shadow-lg text-center max-w-sm w-full">
      <div class="w-16 h-16 bg-purple-100 text-purple-600 rounded-full flex items-center justify-center mx-auto mb-4">
        <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"></path></svg>
      </div>
      <h2 class="text-2xl font-bold text-gray-900 mb-2">Message Sent!</h2>
      <p class="text-gray-600 mb-6">{{ signatureStore.success_message || 'Your direct message was sent successfully.' }}</p>
      
      <div class="space-y-3">
        <button 
            id="back-to-dm-from-dm-success" 
            @click="handleBackDM"
            class="w-full bg-purple-600 text-white font-bold py-2 rounded hover:bg-purple-700"
        >
            Back to Chat
        </button>
        <button 
            id="go-home-from-dm-success" 
            @click="handleGoHome"
            class="w-full text-gray-600 hover:text-gray-900 font-medium"
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
  name: 'START_DM_SUCCESS',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    async function handleBackDM() {
        signatureStore.currentPageId = 'DM_DETAIL'
        await router.push({ name: 'DM_DETAIL', params: { id: signatureStore.selected_dm_id } })
    }

    async function handleGoHome() {
        signatureStore.currentPageId = 'HOME'
        await router.push({ name: 'HOME' })
    }

    return {
        signatureStore,
        handleBackDM,
        handleGoHome
    }
  }
}
</script>