<template>
  <div class="h-screen flex items-center justify-center bg-black/50 p-4">
     <div class="bg-white rounded-lg shadow-xl max-w-md w-full p-6 relative">
         <h2 class="text-xl font-semibold mb-6">Move to folder</h2>
         
         <div class="relative mb-6">
             <div id="move-folder-dropdown" class="w-full border border-gray-300 rounded px-3 py-2 flex justify-between items-center cursor-pointer hover:border-[#0078D4]" @click="toggleDropdown">
                 <span>{{ selectedFolder || 'Select folder' }}</span>
                 <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
             </div>
             
             <div v-if="showDropdown" class="absolute left-0 top-12 w-full bg-white shadow-lg border border-gray-200 rounded z-10">
                 <div id="move-folder-inbox" class="px-4 py-2 hover:bg-gray-100 cursor-pointer" @click="selectFolder('Inbox')">Inbox</div>
                 <div id="move-folder-archive" class="px-4 py-2 hover:bg-gray-100 cursor-pointer" @click="selectFolder('Archive')">Archive</div>
                 <div id="move-folder-junk" class="px-4 py-2 hover:bg-gray-100 cursor-pointer" @click="selectFolder('Junk')">Junk</div>
             </div>
         </div>
         
         <div class="flex justify-end gap-3">
             <button id="move-cancel" class="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded" @click="cancel">Cancel</button>
             <button id="move-confirm-button" class="px-4 py-2 bg-[#0078D4] text-white rounded hover:bg-[#005A9E] disabled:opacity-50" :disabled="!selectedFolder" @click="confirm">Move</button>
         </div>
     </div>
  </div>
</template>

<script>
import { ref } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'MAIL_MOVE',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    
    const showDropdown = ref(false);
    const selectedFolder = ref(null);

    const toggleDropdown = () => {
        showDropdown.value = !showDropdown.value;
    };

    const selectFolder = (folder) => {
        selectedFolder.value = folder;
        showDropdown.value = false;
        signatureStore.handleAction('ACT_MOVE_SELECT_FOLDER', { widget: 'dropdown', folder });
    };

    const confirm = async () => {
        await signatureStore.handleAction('ACT_MOVE_SUBMIT');
        router.push({ name: 'MOVE_EMAIL_SUCCESS' });
    };

    const cancel = async () => {
        await signatureStore.handleAction('ACT_MOVE_BACK_READ');
        router.push({ name: 'MAIL_MESSAGE_READ' });
    };

    return {
        showDropdown,
        selectedFolder,
        toggleDropdown,
        selectFolder,
        confirm,
        cancel
    };
  }
}
</script>