<template>
  <div class="h-screen flex flex-col bg-gray-50 items-center justify-center">
    <div class="bg-white rounded-xl shadow-2xl p-8 max-w-md w-full text-center border-t-4 border-blue-500">
      <div class="w-20 h-20 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-6 animate-pulse">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-10 w-10 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
        </svg>
      </div>
      
      <h2 class="text-2xl font-bold text-gray-800 mb-2">Message Sent!</h2>
      <p class="text-gray-600 mb-8">{{ store.success_message || 'Your message has been delivered.' }}</p>
      
      <button 
        id="chat-message-success-go-home"
        @click="goHome"
        class="w-full bg-[#6264A7] hover:bg-[#464775] text-white font-semibold py-3 px-6 rounded-lg shadow transition-colors"
      >
        Return to Home
      </button>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'CHAT_MESSAGE_SENT_SUCCESS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const goHome = async () => {
      store.success_message = "Message sent";
      store.currentPageId = 'HOME';
      await router.push({ name: 'HOME' });
    }

    return {
      store,
      goHome
    }
  }
}
</script>