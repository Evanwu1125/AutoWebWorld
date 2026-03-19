<template>
  <div class="h-screen flex flex-col bg-white">
    <!-- Header -->
    <header class="bg-[#0078D4] text-white flex items-center h-12 px-4 shadow-md shrink-0 justify-between">
         <div class="flex items-center gap-4">
            <button id="forward-back" class="hover:bg-[#005A9E] p-1 rounded" @click="goBack">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" /></svg>
            </button>
            <span class="font-semibold">Forward</span>
         </div>
         <button id="forward-send-button" class="bg-white text-[#0078D4] px-4 py-1 rounded font-semibold hover:bg-gray-100 flex items-center gap-2" @click="send">
            <span>Send</span>
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path></svg>
         </button>
    </header>

    <div class="flex-1 p-6 max-w-4xl mx-auto w-full flex flex-col gap-4">
        <div class="flex items-center border-b border-gray-200 py-2">
            <label class="w-16 text-gray-500 font-medium">To</label>
            <input type="text" id="forward-to-input" v-model="to" @input="handleInput('to')" class="flex-1 outline-none text-gray-800" placeholder="Recipients" />
        </div>

        <!-- Original Email Context -->
        <div class="bg-gray-50 p-4 border-l-4 border-gray-300 rounded mb-4 text-sm text-gray-600">
            <div class="font-bold">Forwarding: {{ originalEmail?.subject }}</div>
            <div>From: {{ originalEmail?.sender }}</div>
        </div>

        <div class="flex-1 mt-2">
            <textarea id="forward-body-editor" v-model="body" @input="handleInput('body')" class="w-full h-full resize-none outline-none text-gray-800 leading-relaxed font-sans" placeholder="Add message..."></textarea>
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
  name: 'MAIL_FORWARD',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();
    
    const to = ref('');
    const body = ref('');
    const originalEmail = computed(() => dataStore.emails.find(e => e.id === signatureStore.selected_email_id));

    const handleInput = (field) => {
        let val = field === 'to' ? to.value : body.value;
        if (field === 'to') signatureStore.handleAction('ACT_FORWARD_TYPE_TO', { input_text: val, field });
        if (field === 'body') signatureStore.handleAction('ACT_FORWARD_TYPE_BODY', { input_text: val, field });
    };

    const send = async () => {
        await signatureStore.handleAction('ACT_FORWARD_SEND');
        router.push({ name: 'FORWARD_EMAIL_SUCCESS' });
    };

    const goBack = async () => {
        await signatureStore.handleAction('ACT_FORWARD_BACK_READ');
        router.push({ name: 'MAIL_MESSAGE_READ' });
    };

    return {
        to, body,
        originalEmail,
        handleInput,
        send,
        goBack
    };
  }
}
</script>