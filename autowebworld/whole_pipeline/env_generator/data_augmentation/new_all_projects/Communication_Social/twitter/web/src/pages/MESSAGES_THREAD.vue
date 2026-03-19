<template>
  <div class="flex flex-col min-h-screen bg-black text-white">
    <!-- Header -->
    <div class="sticky top-0 z-30 bg-black/80 backdrop-blur-md px-4 py-3 flex items-center gap-4 border-b border-[#2F3336]">
      <div id="thread-back-inbox" @click="handleBackInbox" class="p-2 -ml-2 rounded-full hover:bg-white/10 cursor-pointer transition-colors">
         <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 fill-current"><g><path d="M7.414 13l5.043 5.04-1.414 1.42L3.586 12l7.457-7.46 1.414 1.42L7.414 11H21v2H7.414z"></path></g></svg>
      </div>
      <div class="flex items-center gap-2">
         <h2 class="text-xl font-bold">{{ participant?.name }}</h2>
         <span v-if="participant?.verified">
            <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 text-[#1D9BF0] fill-current"><g><path d="M22.5 12.5c0-1.58-.875-2.95-2.148-3.6.154-.435.238-.905.238-1.4 0-2.21-1.71-3.998-3.818-3.998-.47 0-.92.084-1.336.25C14.818 2.415 13.51 1.5 12 1.5s-2.816.917-3.437 2.25c-.415-.165-.866-.25-1.336-.25-2.11 0-3.818 1.79-3.818 4 0 .495.083.965.238 1.4-1.272.65-2.147 2.018-2.147 3.6 0 1.495.782 2.798 1.942 3.486-.02.17-.032.34-.032.514 0 2.21 1.708 4 3.818 4 .47 0 .92-.086 1.335-.25.62 1.334 1.926 2.25 3.437 2.25 1.512 0 2.818-.916 3.437-2.25.415.163.865.248 1.336.248 2.11 0 3.818-1.79 3.818-4 0-.174-.012-.344-.033-.513 1.158-.687 1.943-1.99 1.943-3.484zm-6.616-3.334l-4.334 6.5c-.145.217-.382.334-.625.334-.143 0-.288-.04-.416-.126l-.115-.094-2.415-2.415c-.293-.293-.293-.768 0-1.06s.768-.294 1.06 0l1.77 1.767 3.825-5.74c.23-.345.696-.436 1.04-.207.346.23.44.696.21 1.04z"></path></g></svg>
         </span>
      </div>
    </div>

    <!-- Messages Area -->
    <div class="flex-1 p-4 overflow-y-auto flex flex-col gap-4 pb-24">
       <!-- Mock Profile at top of chat -->
       <div class="flex flex-col items-center py-6 border-b border-[#2F3336] mb-4">
          <div class="w-16 h-16 rounded-full overflow-hidden bg-gray-700 mb-2">
              <img :src="participant?.avatar || '/images/photo1766328768.jpg'" alt="avatar" class="w-full h-full object-cover">
          </div>
          <div class="font-bold text-lg">{{ participant?.name }}</div>
          <div class="text-[#71767B]">{{ participant?.handle }}</div>
          <div class="text-[#71767B] text-sm mt-2">{{ participant?.bio }}</div>
          <div class="text-[#71767B] text-sm mt-1">Joined {{ participant?.joined_date }}</div>
       </div>

       <!-- Chat Bubbles -->
       <div 
         v-for="msg in threadMessages" 
         :key="msg.id" 
         class="flex flex-col"
         :class="msg.sender_id === 'user_me' ? 'items-end' : 'items-start'"
       >
          <div 
             class="px-4 py-3 rounded-3xl max-w-[80%] text-[15px] break-words"
             :class="msg.sender_id === 'user_me' ? 'bg-[#1D9BF0] text-white rounded-br-none' : 'bg-[#2F3336] text-white rounded-bl-none'"
          >
             {{ msg.text }}
          </div>
          <div class="text-xs text-[#71767B] mt-1 px-1">
             {{ msg.timestamp }}
          </div>
       </div>
    </div>

    <!-- Compose Footer -->
    <div class="sticky bottom-0 bg-black border-t border-[#2F3336] p-3 flex items-center gap-3">
       <div class="p-2 rounded-full hover:bg-[#1D9BF0]/10 text-[#1D9BF0] cursor-pointer">
          <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 fill-current"><g><path d="M19.75 22H4.25C3.01 22 2 20.99 2 19.75V4.25C2 3.01 3.01 2 4.25 2h15.5C20.99 2 22 3.01 22 4.25v15.5c0 1.24-1.01 2.25-2.25 2.25zM4.25 3.5c-.41 0-.75.34-.75.75v15.5c0 .41.34.75.75.75h15.5c.41 0 .75-.34.75-.75V4.25c0-.41-.34-.75-.75-.75H4.25z"></path><path d="M17 9H7v1.5h10V9zm-10 6h10v-1.5H7V15z"></path></g></svg>
       </div>
       <div class="p-2 rounded-full hover:bg-[#1D9BF0]/10 text-[#1D9BF0] cursor-pointer">
          <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 fill-current"><g><path d="M19 10.5V8.8h-4.4v6.4h1.7v-2h2v-1.7h-2v-1h2.7zm-6 0V8.8H8.6v6.4h4.4v-1.7h-2.7v-1h2v-1.7h-2v-1H13zm-7.3 0V8.8H1.3v6.4h1.7v-1.7h2v-1.7h-2v-1.3h2.7z"></path></g></svg>
       </div>
       
       <div id="thread-reply-input" @click="handleReply" class="flex-1 bg-[#202327] rounded-full px-4 py-2 text-[#71767B] cursor-text hover:bg-black border border-transparent focus:border-[#1D9BF0] focus:bg-black transition-colors">
          Start a new message
       </div>
       
       <div class="p-2 rounded-full hover:bg-[#1D9BF0]/10 text-[#1D9BF0] cursor-pointer opacity-50">
           <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 fill-current"><g><path d="M2.504 21.866l.526-2.108C3.04 19.757 4.57 19.9 6.11 19.9H5.6c6.845 0 12.4-5.6 12.4-12.454C18 3.326 14.428.25 10.063.25c-4.366 0-7.938 3.076-7.938 7.2 0 1.54.433 3.063 1.196 4.356L2.49 21.846zM18.25 7.446c0 5.618-4.572 10.204-10.187 10.204-.374 0-.742-.016-1.1-.048l-3.235 1.286.81-3.238C3.93 14.636 3.625 13.38 3.625 12c0-3.315 2.876-5.75 6.438-5.75 3.562 0 6.437 2.435 6.437 5.75V7.446z"></path></g></svg>
       </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';

export default {
  name: 'MESSAGES_THREAD',
  setup() {
    const router = useRouter();
    const route = useRoute();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const threadId = computed(() => route.params.thread_id || signatureStore.thread_id);
    const thread = computed(() => dataStore.getThreadById(threadId.value));
    const participant = computed(() => thread.value ? dataStore.getUserById(thread.value.participant_id) : null);
    
    // Mock messages for this thread
    const threadMessages = computed(() => dataStore.messages.filter(m => m.thread_id === threadId.value));

    const handleBackInbox = () => {
        signatureStore.setCurrentPageId('MESSAGES_INBOX');
        router.push({ name: 'MESSAGES_INBOX' });
    };

    const handleReply = () => {
        // FSM ACT_MESSAGES_THREAD_REPLY -> MESSAGES_COMPOSE
        // Usually reply goes to same thread, but FSM routes to compose. 
        // We'll pre-fill recipient based on thread.
        // Actually FSM doesn't pass thread_id to compose explicitly in parameters, but we should handle context.
        // Let's assume compose can handle "reply to thread X" or we start fresh.
        // ACT_MESSAGES_THREAD_REPLY parameters: thread_id.
        // We will pass this to compose.
        signatureStore.thread_id = threadId.value; 
        signatureStore.setCurrentPageId('MESSAGES_COMPOSE');
        router.push({ name: 'MESSAGES_COMPOSE' });
    };

    return {
        participant,
        threadMessages,
        handleBackInbox,
        handleReply
    };
  }
}
</script>