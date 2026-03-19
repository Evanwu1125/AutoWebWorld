<template>
  <div class="h-screen flex flex-col bg-white">
    <!-- Header -->
    <header class="bg-[#0078D4] text-white flex items-center h-12 px-4 shadow-md shrink-0">
         <button id="back-inbox" class="mr-4 hover:bg-[#005A9E] p-1 rounded" @click="goBackInbox">
             <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" /></svg>
         </button>
         <span class="font-semibold">Reading Pane</span>
    </header>

    <!-- Toolbar -->
    <div class="bg-[#F3F2F1] border-b border-gray-200 p-2 flex gap-2 shrink-0">
        <button id="button-reply" class="flex items-center gap-2 px-3 py-1.5 hover:bg-white hover:shadow-sm rounded transition-colors text-gray-700" @click="handleReply">
             <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 10h10a8 8 0 018 8v2M3 10l6 6m-6-6l6-6"></path></svg>
             <span>Reply</span>
        </button>
        <button id="button-forward" class="flex items-center gap-2 px-3 py-1.5 hover:bg-white hover:shadow-sm rounded transition-colors text-gray-700" @click="handleForward">
             <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path></svg>
             <span>Forward</span>
        </button>
        <div class="w-px bg-gray-300 mx-1"></div>
        <button id="button-move" class="flex items-center gap-2 px-3 py-1.5 hover:bg-white hover:shadow-sm rounded transition-colors text-gray-700" @click="handleMove">
             <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4"></path></svg>
             <span>Move to</span>
        </button>
    </div>

    <!-- Content -->
    <div class="flex-1 p-8 overflow-y-auto" v-if="email">
        <h1 class="text-2xl font-semibold mb-6 text-gray-800">{{ email.subject }}</h1>
        
        <div class="flex items-start gap-4 mb-8">
             <div class="w-12 h-12 rounded-full bg-blue-100 flex items-center justify-center text-[#0078D4] font-bold text-xl">
                 {{ email.sender.charAt(0) }}
             </div>
             <div class="flex-1">
                 <div class="flex justify-between items-baseline">
                     <h3 class="font-bold text-gray-900">{{ email.sender }}</h3>
                     <span class="text-sm text-gray-500">{{ email.time }}</span>
                 </div>
                 <div class="text-sm text-gray-600">To: You</div>
             </div>
        </div>
        
        <div class="prose max-w-none text-gray-800 leading-relaxed whitespace-pre-line">
            {{ email.body }}
        </div>
        
        <!-- Attachment Placeholder -->
        <div v-if="email.hasAttachment" class="mt-8 pt-4 border-t border-gray-100">
             <div class="flex items-center gap-3 p-3 border border-gray-200 rounded-md max-w-xs hover:bg-gray-50 cursor-pointer">
                 <div class="text-red-500 text-2xl">📄</div>
                 <div class="flex-1 min-w-0">
                     <div class="font-medium truncate">Document.pdf</div>
                     <div class="text-xs text-gray-500">245 KB</div>
                 </div>
             </div>
        </div>
    </div>
    
    <div v-else class="flex-1 flex items-center justify-center text-gray-500">
        Loading email...
    </div>
  </div>
</template>

<script>
import { computed, onMounted } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';

export default {
  name: 'MAIL_MESSAGE_READ',
  setup() {
    const route = useRoute();
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const emailId = computed(() => route.params.id || signatureStore.selected_email_id);
    const email = computed(() => dataStore.emails.find(e => e.id === emailId.value));

    onMounted(() => {
        if (!emailId.value) {
            // Fallback or redirect if no email selected
            router.push({ name: 'MAIL_INBOX' });
        }
    });

    const handleReply = async () => {
        await signatureStore.handleAction('ACT_READ_REPLY');
        router.push({ name: 'MAIL_REPLY' });
    };

    const handleForward = async () => {
        await signatureStore.handleAction('ACT_READ_FORWARD');
        router.push({ name: 'MAIL_FORWARD' });
    };

    const handleMove = async () => {
        await signatureStore.handleAction('ACT_READ_MOVE');
        router.push({ name: 'MAIL_MOVE' });
    };

    const goBackInbox = async () => {
        await signatureStore.handleAction('ACT_READ_BACK_INBOX');
        router.push({ name: 'MAIL_INBOX' });
    };

    return {
        email,
        handleReply,
        handleForward,
        handleMove,
        goBackInbox
    };
  }
}
</script>