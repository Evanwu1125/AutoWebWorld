<template>
  <div class="min-h-screen bg-gray-100 flex items-center justify-center p-4">
    <div class="bg-white rounded-xl shadow-lg w-full max-w-lg overflow-hidden flex flex-col h-[500px]">
      <!-- Header -->
      <div class="relative px-4 py-3 border-b border-gray-100 flex items-center justify-center bg-white z-10">
        <div 
          id="message-edit-back"
          @click="goBackEdit"
          class="absolute left-4 top-1/2 transform -translate-y-1/2 p-2 hover:bg-gray-100 rounded-full cursor-pointer transition-colors"
        >
          <svg class="h-6 w-6 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
          </svg>
        </div>
        <h2 class="text-lg font-bold text-gray-900">Review Message</h2>
        <div 
          id="message-cancel-from-review" 
          @click="cancelReview"
          class="absolute right-4 top-1/2 transform -translate-y-1/2 p-1 text-gray-500 hover:bg-gray-100 rounded-full cursor-pointer transition-colors"
        >
          Cancel
        </div>
      </div>

      <!-- Content -->
      <div class="flex-1 p-6 flex flex-col">
        <div class="mb-6">
          <span class="text-xs font-bold text-gray-500 uppercase tracking-wide">To</span>
          <div class="mt-1 flex items-center gap-2">
            <div class="h-6 w-6 rounded-full bg-blue-100 flex items-center justify-center text-blue-600 font-bold text-xs">U</div>
            <span class="text-gray-900 font-medium">Selected User</span>
          </div>
        </div>

        <div class="flex-1">
          <span class="text-xs font-bold text-gray-500 uppercase tracking-wide">Message</span>
          <div class="mt-2 bg-gray-50 rounded-xl p-4 border border-gray-200 text-gray-900 leading-relaxed">
            {{ messageText }}
          </div>
        </div>

        <div class="mt-auto bg-yellow-50 text-yellow-800 text-xs p-3 rounded-md flex items-start gap-2">
          <span>⚠️</span>
          <p>Please ensure your message follows our community guidelines. This message will be sent immediately.</p>
        </div>
      </div>

      <!-- Footer -->
      <div class="p-4 border-t border-gray-100 bg-gray-50">
        <button 
          id="message-send-button"
          @click="sendMessage"
          class="w-full py-2 bg-blue-600 text-white font-semibold rounded-full shadow-sm hover:bg-blue-700 transition-colors flex items-center justify-center gap-2"
        >
          Send Message
          <svg class="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
            <path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z"/>
          </svg>
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';

export default {
  name: 'MESSAGE_REVIEW',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();
    
    const messageText = computed(() => signatureStore.message_text);

    const sendMessage = async () => {
      // Mock sending logic (add to threads if needed, but for now just navigate)
      signatureStore.currentPageId = 'MESSAGE_SEND_SUCCESS';
      await router.push({ name: 'MESSAGE_SEND_SUCCESS' });
    };

    const goBackEdit = async () => {
      signatureStore.currentPageId = 'MESSAGE_COMPOSE';
      await router.push({ name: 'MESSAGE_COMPOSE' });
    };

    const cancelReview = async () => {
      // Clear
      signatureStore.message_text = null;
      signatureStore.recipient_selected = null;
      signatureStore.currentPageId = 'MESSENGER_INBOX';
      await router.push({ name: 'MESSENGER_INBOX' });
    };

    return {
      messageText,
      sendMessage,
      goBackEdit,
      cancelReview
    };
  }
}
</script>