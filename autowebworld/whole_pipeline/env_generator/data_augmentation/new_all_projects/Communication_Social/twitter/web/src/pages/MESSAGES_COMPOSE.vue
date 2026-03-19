<template>
  <div class="flex flex-col min-h-screen bg-black text-white">
    <!-- Header -->
    <div class="sticky top-0 z-30 bg-black/80 backdrop-blur-md px-4 py-3 flex items-center justify-between border-b border-[#2F3336]">
      <div class="flex items-center gap-4">
        <div id="message-compose-back" @click="handleBackInbox" class="p-2 -ml-2 rounded-full hover:bg-white/10 cursor-pointer transition-colors">
            <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 fill-current"><g><path d="M10.59 12L4.54 5.96l1.42-1.42L12 10.59l6.04-6.05 1.42 1.42L13.41 12l6.05 6.04-1.42 1.42L12 13.41l-6.04 6.05-1.42-1.42L10.59 12z"></path></g></svg>
        </div>
        <h2 class="text-xl font-bold">New message</h2>
      </div>
      <button 
         id="message-send-button" 
         @click="handleSend"
         :disabled="!isValid"
         :class="isValid ? 'bg-white text-black hover:bg-[#EFF3F4]' : 'bg-[#787a7a] text-[#16181C] cursor-not-allowed'"
         class="font-bold rounded-full px-4 py-1.5 transition-colors"
      >
        Next
      </button>
    </div>

    <!-- Recipient Search -->
    <div class="p-4 border-b border-[#2F3336] flex items-center gap-2">
       <div class="text-[#1D9BF0] flex-shrink-0">
          <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 fill-current"><g><path d="M10.25 3.75c-3.59 0-6.5 2.91-6.5 6.5s2.91 6.5 6.5 6.5c1.795 0 3.419-.726 4.596-1.904 1.178-1.177 1.904-2.801 1.904-4.596 0-3.59-2.91-6.5-6.5-6.5zm-8.5 6.5c0-4.694 3.806-8.5 8.5-8.5s8.5 3.806 8.5 8.5c0 1.986-.73 3.815-1.945 5.232l4.944 4.942-1.414 1.415-4.942-4.944C14.065 18.02 12.236 18.75 10.25 18.75c-4.694 0-8.5-3.806-8.5-8.5z"></path></g></svg>
       </div>
       <input 
          id="message-recipient-input"
          v-model="recipientQuery"
          @input="handleRecipientInput"
          type="text" 
          placeholder="Search people" 
          class="w-full bg-transparent text-white focus:outline-none placeholder-gray-500"
       >
    </div>

    <!-- Suggested Users (Visual Filler) -->
    <div class="p-4">
       <div v-if="!recipientQuery" class="text-sm font-bold text-[#71767B] mb-2">Suggested</div>
       <div v-if="!recipientQuery" class="flex flex-col gap-4">
          <!-- Mock suggested users -->
          <div class="flex items-center gap-3">
              <div class="w-10 h-10 rounded-full bg-gray-700"></div>
              <div class="flex flex-col">
                  <div class="font-bold">Elon Musk</div>
                  <div class="text-[#71767B]">@elonmusk</div>
              </div>
          </div>
       </div>
    </div>
    
    <!-- Message Body (FSM implies typing text too in same page? "ACT_MESSAGES_COMPOSE_TYPE_TEXT") -->
    <!-- Typically "Next" leads to chat thread where you type. But FSM has ACT_MESSAGES_COMPOSE_SEND in this page.
         So we need a text area here too, typically at bottom. -->
    <div class="flex-1"></div>
    
    <div class="p-3 border-t border-[#2F3336] sticky bottom-0 bg-black">
       <div class="bg-[#202327] rounded-2xl flex items-center px-4 py-2">
           <textarea 
              id="message-textarea"
              v-model="messageText"
              @input="handleMessageInput"
              placeholder="Start a new message"
              class="w-full bg-transparent text-white focus:outline-none resize-none h-10 py-2"
           ></textarea>
           <div @click="handleSend" class="p-2 -mr-2 rounded-full hover:bg-[#1D9BF0]/10 text-[#1D9BF0] cursor-pointer" :class="!isValid ? 'opacity-50' : ''">
               <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 fill-current"><g><path d="M2.504 21.866l.526-2.108C3.04 19.757 4.57 19.9 6.11 19.9H5.6c6.845 0 12.4-5.6 12.4-12.454C18 3.326 14.428.25 10.063.25c-4.366 0-7.938 3.076-7.938 7.2 0 1.54.433 3.063 1.196 4.356L2.49 21.846zM18.25 7.446c0 5.618-4.572 10.204-10.187 10.204-.374 0-.742-.016-1.1-.048l-3.235 1.286.81-3.238C3.93 14.636 3.625 13.38 3.625 12c0-3.315 2.876-5.75 6.438-5.75 3.562 0 6.437 2.435 6.437 5.75V7.446z"></path></g></svg>
           </div>
       </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'MESSAGES_COMPOSE',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    
    const recipientQuery = ref('');
    const messageText = ref('');
    
    // Preconditions check: recipient > 0, message > 0
    const isValid = computed(() => recipientQuery.value.length > 0 && messageText.value.length > 0);

    const handleRecipientInput = () => {
        signatureStore.recipient_user_id = recipientQuery.value; // Store the input string as ID/Query for now
    };

    const handleMessageInput = () => {
        signatureStore.message_text = messageText.value;
    };

    const handleSend = () => {
        if (!isValid.value) return;
        signatureStore.setCurrentPageId('MESSAGE_SEND_SUCCESS');
        router.push({ name: 'MESSAGE_SEND_SUCCESS' });
    };

    const handleBackInbox = () => {
        signatureStore.setCurrentPageId('MESSAGES_INBOX');
        router.push({ name: 'MESSAGES_INBOX' });
    };

    return {
        recipientQuery,
        messageText,
        isValid,
        handleRecipientInput,
        handleMessageInput,
        handleSend,
        handleBackInbox
    };
  }
}
</script>