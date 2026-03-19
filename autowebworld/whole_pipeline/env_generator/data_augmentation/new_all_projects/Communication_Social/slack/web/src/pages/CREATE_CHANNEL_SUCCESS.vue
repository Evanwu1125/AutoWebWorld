<template>
  <div class="h-screen flex flex-col items-center justify-center bg-blue-50">
    <div class="bg-white p-8 rounded-xl shadow-lg text-center max-w-sm w-full">
      <div class="w-16 h-16 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center mx-auto mb-4">
        <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"></path></svg>
      </div>
      <h2 class="text-2xl font-bold text-gray-900 mb-2">Channel Saved!</h2>
      <p class="text-gray-600 mb-6">{{ signatureStore.success_message || 'Channel settings have been updated.' }}</p>
      
      <div class="space-y-3">
        <button 
            id="back-to-channels-from-channel-success" 
            @click="handleBackList"
            class="w-full bg-blue-600 text-white font-bold py-2 rounded hover:bg-blue-700"
        >
            Back to Channels
        </button>
        <button 
            id="go-home-from-channel-success" 
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
  name: 'CREATE_CHANNEL_SUCCESS',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    async function handleBackList() {
        signatureStore.currentPageId = 'CHANNEL_LIST'
        await router.push({ name: 'CHANNEL_LIST' })
    }

    async function handleGoHome() {
        signatureStore.currentPageId = 'HOME'
        await router.push({ name: 'HOME' })
    }

    return {
        signatureStore,
        handleBackList,
        handleGoHome
    }
  }
}
</script>