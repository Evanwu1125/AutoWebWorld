<template>
  <div class="min-h-screen bg-gray-100 flex items-center justify-center p-4">
    <div class="bg-white rounded-xl shadow-lg w-full max-w-sm text-center p-8">
      <div class="mx-auto flex items-center justify-center h-16 w-16 rounded-full bg-blue-100 mb-6">
        <svg class="h-8 w-8 text-blue-600 transform rotate-90" fill="currentColor" viewBox="0 0 20 20">
            <path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z"/>
        </svg>
      </div>
      
      <h2 class="text-2xl font-bold text-gray-900 mb-2">Message Sent</h2>
      <p class="text-gray-600 mb-8">Your message has been delivered.</p>
      
      <div class="space-y-3">
        <button 
          id="message-success-view-inbox"
          @click="viewInbox"
          class="w-full py-2 bg-blue-600 text-white font-semibold rounded-md shadow-sm hover:bg-blue-700 transition-colors"
        >
          Back to Inbox
        </button>
        <button 
          id="message-success-home"
          @click="goHome"
          class="w-full py-2 bg-white text-gray-700 font-semibold rounded-md border border-gray-300 hover:bg-gray-50 transition-colors"
        >
          Go Home
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'MESSAGE_SEND_SUCCESS',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();

    const viewInbox = async () => {
      // Clear compose state
      signatureStore.message_text = null;
      signatureStore.recipient_selected = null;
      
      signatureStore.currentPageId = 'MESSENGER_INBOX';
      await router.push({ name: 'MESSENGER_INBOX' });
    };

    const goHome = async () => {
      signatureStore.success_message = "Message sent"; // FSM Effect
      // Clear compose state
      signatureStore.message_text = null;
      signatureStore.recipient_selected = null;
      
      signatureStore.currentPageId = 'HOME';
      await router.push({ name: 'HOME' });
    };

    return {
      viewInbox,
      goHome
    };
  }
}
</script>