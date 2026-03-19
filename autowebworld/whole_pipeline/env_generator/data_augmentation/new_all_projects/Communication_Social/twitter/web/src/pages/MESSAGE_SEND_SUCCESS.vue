<template>
  <div class="flex flex-col items-center justify-center min-h-screen bg-black text-white p-6 text-center">
    <div class="w-16 h-16 bg-[#1D9BF0] rounded-full flex items-center justify-center mb-6">
       <svg viewBox="0 0 24 24" aria-hidden="true" class="h-8 w-8 fill-white"><g><path d="M1.998 5.5c0-1.381 1.119-2.5 2.5-2.5h15c1.381 0 2.5 1.119 2.5 2.5v13c0 1.381-1.119 2.5-2.5 2.5h-15c-1.381 0-2.5-1.119-2.5-2.5v-13zm2.5-.5c-.276 0-.5.224-.5.5v2.764l8 3.638 8-3.636V5.5c0-.276-.224-.5-.5-.5h-15zm15.5 5.463l-8 3.636-8-3.638V18.5c0 .276.224.5.5.5h15c.276 0 .5-.224.5-.5v-8.037z"></path></g></svg>
    </div>
    
    <h2 class="text-2xl font-bold mb-2">Message Sent!</h2>
    <p class="text-[#71767B] mb-8">Your message has been sent successfully.</p>
    
    <div class="flex flex-col gap-4 w-full max-w-xs">
       <button id="message-success-view-inbox" @click="handleViewInbox" class="w-full border border-[#536471] text-white font-bold rounded-full py-3 hover:bg-white/10 transition-colors">
          View Inbox
       </button>
       <button id="message-success-go-home" @click="handleGoHome" class="w-full bg-white text-black font-bold rounded-full py-3 hover:bg-[#EFF3F4] transition-colors">
          Back to Home
       </button>
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

    const handleViewInbox = () => {
       signatureStore.setCurrentPageId('MESSAGES_INBOX');
       router.push({ name: 'MESSAGES_INBOX' });
    };

    const handleGoHome = () => {
       signatureStore.setCurrentPageId('HOME');
       router.push({ name: 'HOME' });
    };

    return {
       handleViewInbox,
       handleGoHome
    };
  }
}
</script>