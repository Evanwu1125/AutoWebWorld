<template>
  <div class="h-screen flex flex-col items-center justify-center bg-green-50">
    <div class="bg-white p-8 rounded-xl shadow-lg text-center max-w-sm w-full">
      <div class="w-16 h-16 bg-green-100 text-green-600 rounded-full flex items-center justify-center mx-auto mb-4">
        <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
      </div>
      <h2 class="text-2xl font-bold text-gray-900 mb-2">Message Sent!</h2>
      <p class="text-gray-600 mb-6">{{ signatureStore.success_message || 'Your message has been delivered.' }}</p>
      
      <div class="space-y-3">
        <button 
            id="back-to-channel-from-send-success" 
            @click="handleBackChannel"
            class="w-full bg-green-600 text-white font-bold py-2 rounded hover:bg-green-700"
        >
            Back to Channel
        </button>
        <button 
            id="go-home-from-send-success" 
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
  name: 'SEND_MESSAGE_SUCCESS',
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