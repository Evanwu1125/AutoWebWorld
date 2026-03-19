<template>
  <div class="h-screen flex flex-col bg-white">
    <!-- Header -->
    <header class="bg-[#0078D4] text-white flex items-center h-12 px-4 shadow-md shrink-0 justify-between">
         <div class="flex items-center gap-4">
            <button id="compose-discard" class="hover:bg-[#005A9E] p-1 rounded" @click="discard">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" /></svg>
            </button>
            <span class="font-semibold">New Message</span>
         </div>
         <button id="compose-send-button" class="bg-white text-[#0078D4] px-4 py-1 rounded font-semibold hover:bg-gray-100 flex items-center gap-2" @click="send">
            <span>Send</span>
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path></svg>
         </button>
    </header>

    <div class="flex-1 p-6 max-w-4xl mx-auto w-full flex flex-col gap-4">
        <div class="flex items-center border-b border-gray-200 py-2">
            <label class="w-16 text-gray-500 font-medium">To</label>
            <input type="text" id="compose-to-input" v-model="to" @input="handleInput('to')" class="flex-1 outline-none text-gray-800" placeholder="Recipients" />
        </div>
        
        <div class="flex items-center border-b border-gray-200 py-2">
            <label class="w-16 text-gray-500 font-medium">Subject</label>
            <input type="text" id="compose-subject-input" v-model="subject" @input="handleInput('subject')" class="flex-1 outline-none text-gray-800 font-medium" placeholder="Add a subject" />
        </div>
        
        <div class="flex-1 mt-4">
            <textarea id="compose-body-editor" v-model="body" @input="handleInput('body')" class="w-full h-full resize-none outline-none text-gray-800 leading-relaxed font-sans" placeholder="Type your message here..."></textarea>
        </div>
        
        <!-- Toolbar Bottom -->
        <div class="flex gap-4 border-t border-gray-100 pt-4 text-gray-500">
            <button class="hover:text-[#0078D4]">A</button>
            <button class="hover:text-[#0078D4]">📎</button>
            <button class="hover:text-[#0078D4]">📷</button>
            <button class="hover:text-[#0078D4]">😊</button>
        </div>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'MAIL_COMPOSE',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    
    const to = ref('');
    const subject = ref('');
    const body = ref('');

    const handleInput = (field) => {
        let val = '';
        if (field === 'to') val = to.value;
        if (field === 'subject') val = subject.value;
        if (field === 'body') val = body.value;
        
        // Map to specific action based on field
        if (field === 'to') signatureStore.handleAction('ACT_COMPOSE_TYPE_TO', { input_text: val, field });
        if (field === 'subject') signatureStore.handleAction('ACT_COMPOSE_TYPE_SUBJECT', { input_text: val, field });
        if (field === 'body') signatureStore.handleAction('ACT_COMPOSE_TYPE_BODY', { input_text: val, field });
    };

    const send = async () => {
        await signatureStore.handleAction('ACT_COMPOSE_SEND');
        router.push({ name: 'SEND_EMAIL_SUCCESS' });
    };

    const discard = async () => {
        await signatureStore.handleAction('ACT_COMPOSE_BACK_INBOX');
        router.push({ name: 'MAIL_INBOX' });
    };

    return {
        to, subject, body,
        handleInput,
        send,
        discard
    };
  }
}
</script>