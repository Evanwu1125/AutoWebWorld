<template>
  <div class="h-screen flex flex-col items-center justify-center bg-[#F0F2F5] p-4 text-center">
      <div class="w-24 h-24 bg-gray-200 rounded-full flex items-center justify-center mb-6">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-12 w-12 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" /></svg>
      </div>
      <h1 class="text-3xl font-bold text-gray-900 mb-2">Moved Successfully</h1>
      <p class="text-gray-600 mb-8 max-w-md">The item has been moved to the selected folder.</p>
      
      <div class="flex gap-4">
          <button id="move-success-home" class="px-6 py-2 border border-gray-300 bg-white rounded-md hover:bg-gray-50 font-medium text-gray-700 transition-colors" @click="goHome">
              Back to Home
          </button>
          <button id="move-success-inbox" class="px-6 py-2 bg-[#0078D4] text-white rounded-md hover:bg-[#005A9E] font-medium transition-colors shadow-sm" @click="goInbox">
              Go to Inbox
          </button>
      </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'MOVE_EMAIL_SUCCESS',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();

    const goHome = async () => {
        await signatureStore.handleAction('ACT_MOVE_SUCCESS_GO_HOME');
        router.push({ name: 'HOME' });
    };

    const goInbox = async () => {
        await signatureStore.handleAction('ACT_MOVE_SUCCESS_GO_INBOX');
        router.push({ name: 'MAIL_INBOX' });
    };

    return { goHome, goInbox };
  }
}
</script>