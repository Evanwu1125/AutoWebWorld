<template>
  <div class="h-screen flex flex-col bg-white overflow-hidden">
    <!-- Header -->
    <header class="bg-[#0078D4] text-white flex items-center h-12 px-4 shadow-md z-20 shrink-0">
        <button id="people-back-home" class="mr-4 hover:bg-[#005A9E] p-1 rounded" @click="goHome">
             <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" /></svg>
        </button>
        <div class="font-semibold">People</div>
    </header>

    <div class="flex flex-1 overflow-hidden">
        <!-- Sidebar -->
        <div class="w-64 bg-gray-50 border-r border-gray-200 hidden md:block">
            <div class="p-4">
                <div class="font-bold text-gray-700 mb-2">My Contacts</div>
                <div class="bg-blue-100 text-[#0078D4] px-4 py-2 rounded-md font-medium cursor-pointer">All Contacts</div>
                <div class="px-4 py-2 hover:bg-gray-100 cursor-pointer text-gray-600">Company Directory</div>
                <div class="px-4 py-2 hover:bg-gray-100 cursor-pointer text-gray-600">Imported</div>
            </div>
        </div>

        <!-- Contact List -->
        <div class="flex-1 flex flex-col">
            <div class="p-4 border-b border-gray-200 bg-white">
                <h2 class="text-xl font-semibold text-gray-800">All Contacts</h2>
            </div>
            
            <div id="people-list" class="flex-1 overflow-y-auto p-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4" @scroll="handleScroll">
                 <div v-for="contact in contacts" :key="contact.id" 
                      :class="['bg-white border border-gray-200 rounded-lg p-4 flex items-center gap-4 hover:shadow-md cursor-pointer transition-shadow group',
                               'row-visible',
                               `data-id-${contact.id}`]"
                      @click="openContact(contact)">
                      
                      <div class="w-12 h-12 rounded-full flex items-center justify-center text-white font-bold text-lg" 
                           :class="getAvatarColor(contact.name)">
                          {{ getInitials(contact.name) }}
                      </div>
                      
                      <div class="min-w-0">
                          <div class="font-semibold text-gray-900 truncate group-hover:text-[#0078D4]">{{ contact.name }}</div>
                          <div class="text-sm text-gray-500 truncate">{{ contact.email }}</div>
                          <div class="text-xs text-gray-400 mt-1">{{ contact.jobTitle }}</div>
                      </div>
                 </div>
            </div>
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
  name: 'PEOPLE_LIST',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();
    
    const contacts = computed(() => dataStore.contacts || []);

    const openContact = async (contact) => {
        // Precondition people_viewport_anchor_id > 0
        signatureStore.people_viewport_anchor_id = contact.id;
        await signatureStore.handleAction('ACT_PEOPLE_OPEN_CONTACT', { item_id: contact.id });
        router.push({ name: 'PEOPLE_DETAIL', params: { id: contact.id } });
    };

    const handleScroll = () => {
        if (contacts.value.length > 0) {
            signatureStore.handleAction('ACT_PEOPLE_SCROLL', { item_id: contacts.value[0].id });
        }
    };

    const goHome = async () => {
        await signatureStore.handleAction('ACT_PEOPLE_BACK_HOME');
        router.push({ name: 'HOME' });
    };

    // Helpers
    const getInitials = (name) => {
        return name.split(' ').map(n => n[0]).join('').substring(0, 2).toUpperCase();
    };

    const getAvatarColor = (name) => {
        const colors = ['bg-blue-500', 'bg-green-500', 'bg-yellow-500', 'bg-purple-500', 'bg-red-500', 'bg-indigo-500'];
        let hash = 0;
        for (let i = 0; i < name.length; i++) {
            hash = name.charCodeAt(i) + ((hash << 5) - hash);
        }
        return colors[Math.abs(hash) % colors.length];
    };

    return {
        contacts,
        openContact,
        handleScroll,
        goHome,
        getInitials,
        getAvatarColor
    };
  }
}
</script>