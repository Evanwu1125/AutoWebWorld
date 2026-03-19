<template>
  <div class="h-screen flex flex-col items-center justify-center bg-yellow-50">
    <div class="bg-white p-8 rounded-xl shadow-lg text-center max-w-sm w-full">
      <div class="w-16 h-16 bg-yellow-100 text-yellow-600 rounded-full flex items-center justify-center mx-auto mb-4">
        <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
      </div>
      <h2 class="text-2xl font-bold text-gray-900 mb-2">Scheduled!</h2>
      <p class="text-gray-600 mb-6">{{ signatureStore.success_message || 'Message scheduled for later.' }}</p>
      
      <div class="space-y-3">
        <button 
            id="back-to-channel-from-schedule-success" 
            @click="handleBackChannel"
            class="w-full bg-yellow-600 text-white font-bold py-2 rounded hover:bg-yellow-700"
        >
            Back to Channel
        </button>
        <button 
            id="go-home-from-schedule-success" 
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
  name: 'SCHEDULE_MESSAGE_SUCCESS',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    async function handleBackChannel() {
        signatureStore.currentPageId = 'CHANNEL_DETAIL'
        await router.push({ name: 'CHANNEL_DETAIL', params: { id: signatureStore.selected_channel_id } })
    }

    async function handleGoHome() {
        signatureStore.currentPageId = 'HOME'
        await router.push({ name: 'HOME' })
    }

    return {
        signatureStore,
        handleBackChannel,
        handleGoHome
    }
  }
}
</script>