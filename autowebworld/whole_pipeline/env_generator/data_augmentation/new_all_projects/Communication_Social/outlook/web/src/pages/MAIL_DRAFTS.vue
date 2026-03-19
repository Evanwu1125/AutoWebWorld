<template>
  <div class="h-screen flex flex-col bg-white overflow-hidden">
    <!-- Header -->
    <header class="bg-[#0078D4] text-white flex items-center h-12 px-4 shadow-md z-20 shrink-0">
        <button id="drafts-back-inbox" class="mr-4 hover:bg-[#005A9E] p-1 rounded" @click="goBackInbox">
             <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" /></svg>
        </button>
        <div class="font-semibold">Drafts</div>
    </header>

    <div class="flex flex-1 overflow-hidden">
        <!-- Drafts List -->
        <div class="w-full md:w-96 border-r border-gray-200 flex flex-col bg-white">
            <div id="drafts-list" class="flex-1 overflow-y-auto" @scroll="handleScroll">
                 <div v-for="email in displayEmails" :key="email.id" 
                      :class="['p-4 border-b border-gray-100 cursor-pointer hover:bg-[#F3F2F1] group',
                               'row-visible',
                               `data-id-${email.id}`]"
                      @click="openDraft(email)">
                      
                      <div class="flex justify-between items-start mb-1">
                          <div class="font-semibold text-red-500 text-sm">[Draft]</div>
                          <div class="text-xs text-gray-500 whitespace-nowrap">{{ email.time }}</div>
                      </div>
                      <div class="text-[#0078D4] text-sm font-medium mb-1 truncate">{{ email.subject || '(No subject)' }}</div>
                      <div class="text-gray-500 text-xs truncate">{{ email.preview }}</div>
                 </div>
            </div>
        </div>

        <div class="hidden md:flex flex-1 bg-gray-50 items-center justify-center text-gray-400">
            <div>Select a draft to continue editing</div>
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
  name: 'MAIL_DRAFTS',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();
    
    const displayEmails = computed(() => dataStore.draftEmails || []);

    const openDraft = async (email) => {
        // Precondition: drafts_viewport_anchor_id > 0
        // We set it manually here to satisfy precondition if scroll didn't happen
        signatureStore.drafts_viewport_anchor_id = email.id;
        await signatureStore.handleAction('ACT_DRAFTS_OPEN_ANY', { item_id: email.id });
        router.push({ name: 'MAIL_COMPOSE' }); // Assuming opens in compose
    };

    const handleScroll = () => {
         if (displayEmails.value.length > 0) {
             signatureStore.handleAction('ACT_DRAFTS_SCROLL_INTO_VIEW', { item_id: displayEmails.value[0].id });
         }
    };

    const goBackInbox = async () => {
        await signatureStore.handleAction('ACT_DRAFTS_BACK_INBOX');
        router.push({ name: 'MAIL_INBOX' });
    };

    return {
        displayEmails,
        openDraft,
        handleScroll,
        goBackInbox
    };
  }
}
</script>