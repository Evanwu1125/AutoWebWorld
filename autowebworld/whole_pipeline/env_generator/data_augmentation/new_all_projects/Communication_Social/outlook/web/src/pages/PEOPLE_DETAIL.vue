<template>
  <div class="h-screen flex flex-col bg-white">
    <!-- Header -->
    <header class="bg-[#0078D4] text-white flex items-center h-12 px-4 shadow-md shrink-0">
        <button id="contact-back-list" class="mr-4 hover:bg-[#005A9E] p-1 rounded" @click="goBackList">
             <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" /></svg>
        </button>
        <div class="font-semibold">Contact Details</div>
    </header>

    <div class="flex-1 p-8 max-w-4xl mx-auto w-full" v-if="contact">
        <!-- Hero Profile -->
        <div class="flex flex-col md:flex-row items-center md:items-start gap-8 mb-12">
             <div class="w-32 h-32 rounded-full bg-blue-100 flex items-center justify-center text-[#0078D4] text-4xl font-bold shadow-inner">
                 {{ getInitials(contact.name) }}
             </div>
             <div class="text-center md:text-left">
                 <h1 class="text-3xl font-bold text-gray-900 mb-2">{{ contact.name }}</h1>
                 <p class="text-lg text-gray-600 mb-1">{{ contact.jobTitle }}</p>
                 <p class="text-gray-500 mb-6">{{ contact.company }}</p>
                 
                 <div class="flex gap-4 justify-center md:justify-start">
                     <button id="contact-send-mail" class="bg-[#0078D4] text-white px-6 py-2 rounded shadow-sm hover:bg-[#005A9E] flex items-center gap-2" @click="sendMail">
                         <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path d="M2.003 5.884L10 9.882l7.997-3.998A2 2 0 0016 4H4a2 2 0 00-1.997 1.884z" /><path d="M18 8.118l-8 4-8-4V14a2 2 0 002 2h12a2 2 0 002-2V8.118z" /></svg>
                         Send Email
                     </button>
                     <button class="bg-white border border-gray-300 text-gray-700 px-6 py-2 rounded shadow-sm hover:bg-gray-50">
                         Schedule Meeting
                     </button>
                 </div>
             </div>
        </div>
        
        <!-- Info Grid -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-8 border-t border-gray-100 pt-8">
             <div class="space-y-6">
                 <div>
                     <label class="block text-sm font-medium text-gray-500 uppercase tracking-wider mb-1">Email</label>
                     <div class="text-gray-900 hover:text-[#0078D4] cursor-pointer">{{ contact.email }}</div>
                 </div>
                 <div>
                     <label class="block text-sm font-medium text-gray-500 uppercase tracking-wider mb-1">Work Phone</label>
                     <div class="text-gray-900">{{ contact.phone }}</div>
                 </div>
             </div>
             
             <div class="space-y-6">
                 <div>
                     <label class="block text-sm font-medium text-gray-500 uppercase tracking-wider mb-1">Department</label>
                     <div class="text-gray-900">{{ contact.department }}</div>
                 </div>
                 <div>
                     <label class="block text-sm font-medium text-gray-500 uppercase tracking-wider mb-1">Location</label>
                     <div class="text-gray-900">{{ contact.location }}</div>
                 </div>
             </div>
        </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';

export default {
  name: 'PEOPLE_DETAIL',
  setup() {
    const route = useRoute();
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();
    
    // FSM might store selected contact ID or we use route param.
    // For simplicity, using route param which maps to FSM logic usually.
    // Assuming FSM sets selected_contact_id somewhere but PEOPLE_LIST doesn't have an effect for it in provided JSON?
    // Checking JSON... Ah, PEOPLE_LIST has no effect to set selected_contact_id!
    // But PEOPLE_DETAIL signature has it.
    // It's likely intended to be passed via URL or implied.
    // We'll rely on route param and mock data lookup.
    
    const contactId = computed(() => route.params.id);
    const contact = computed(() => dataStore.contacts.find(c => c.id === contactId.value));

    const sendMail = async () => {
        await signatureStore.handleAction('ACT_PEOPLE_DETAIL_SEND_MAIL');
        router.push({ name: 'MAIL_COMPOSE' });
    };

    const goBackList = async () => {
        await signatureStore.handleAction('ACT_PEOPLE_DETAIL_BACK_LIST');
        router.push({ name: 'PEOPLE_LIST' });
    };

    const getInitials = (name) => {
        return name ? name.split(' ').map(n => n[0]).join('').substring(0, 2).toUpperCase() : '';
    };

    return {
        contact,
        sendMail,
        goBackList,
        getInitials
    };
  }
}
</script>