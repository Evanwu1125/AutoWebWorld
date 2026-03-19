<template>
  <div class="h-screen flex flex-col bg-white overflow-hidden">
    <!-- Header -->
    <header class="bg-[#0078D4] text-white flex items-center h-12 px-4 shadow-md z-20 shrink-0">
        <button id="trash-back-inbox" class="mr-4 hover:bg-[#005A9E] p-1 rounded" @click="goBackInbox">
             <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" /></svg>
        </button>
        <div class="font-semibold">Deleted Items</div>
    </header>

    <div class="flex flex-1 overflow-hidden">
        <!-- Trash List -->
        <div class="w-full md:w-96 border-r border-gray-200 flex flex-col bg-white">
            <!-- Toolbar -->
            <div class="p-3 border-b border-gray-200 text-sm">
                <label class="flex items-center gap-1 cursor-pointer select-none">
                    <input type="checkbox" id="trash-filter-deleted-today" @change="handleFilterCheckbox" class="rounded text-[#0078D4] focus:ring-[#0078D4]" />
                    <span>Deleted Today</span>
                </label>
            </div>

            <!-- Email Items -->
            <div id="trash-list" class="flex-1 overflow-y-auto" @scroll="handleScroll">
                 <div v-for="email in displayEmails" :key="email.id" 
                      :class="['p-4 border-b border-gray-100 cursor-pointer hover:bg-[#F3F2F1]',
                               isFiltered ? 'row-filtered' : '',
                               `data-id-${email.id}`]"
                      @click="openEmail(email)">
                      
                      <div class="flex justify-between items-start mb-1">
                          <div class="font-semibold text-gray-900 truncate pr-2">{{ email.sender }}</div>
                          <div class="text-xs text-gray-500 whitespace-nowrap">{{ email.time }}</div>
                      </div>
                      <div class="text-[#0078D4] text-sm font-medium mb-1 truncate">{{ email.subject }}</div>
                 </div>
            </div>
        </div>

        <div class="hidden md:flex flex-1 bg-gray-50 items-center justify-center text-gray-400">
            <div>Select an item to view</div>
        </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';

export default {
  name: 'MAIL_TRASH',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();
    
    const isFiltered = ref(false);

    const displayEmails = computed(() => {
        let emails = dataStore.trashEmails || [];
        if (isFiltered.value) {
            // Mock logic for "Deleted Today"
            const today = new Date().toDateString();
            emails = emails.filter(e => new Date(e.date).toDateString() === today);
        }
        return emails;
    });

    const handleFilterCheckbox = (e) => {
        isFiltered.value = e.target.checked;
        signatureStore.handleAction('ACT_TRASH_FILTER_CHECKBOX', { widget: 'checkboxes' });
    };

    const openEmail = async (email) => {
        if (isFiltered.value) {
             await signatureStore.handleAction('ACT_TRASH_OPEN_FILTERED_EMAIL', { item_id: email.id });
        }
        router.push({ name: 'MAIL_MESSAGE_READ', params: { id: email.id } });
    };

    const handleScroll = () => {
         if (displayEmails.value.length > 0) {
             signatureStore.handleAction('ACT_TRASH_SCROLL_INTO_VIEW', { item_id: displayEmails.value[0].id });
         }
    };

    const goBackInbox = async () => {
        await signatureStore.handleAction('ACT_TRASH_BACK_INBOX');
        router.push({ name: 'MAIL_INBOX' });
    };

    return {
        isFiltered,
        displayEmails,
        handleFilterCheckbox,
        openEmail,
        handleScroll,
        goBackInbox
    };
  }
}
</script>