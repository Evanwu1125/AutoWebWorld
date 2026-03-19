<template>
  <div class="h-screen bg-slate-50 flex flex-col">
    <header class="bg-white shadow-sm z-20 px-4 py-3 flex items-center">
        <button id="contact-back-list" @click="goBackList" class="p-2 text-slate-500 hover:text-blue-600 mr-4">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
        </button>
        <h1 class="text-xl font-bold text-slate-800">Contact Info</h1>
    </header>

    <div class="flex-1 overflow-y-auto p-4">
        <div class="max-w-md mx-auto space-y-6">
            <!-- Profile Card -->
            <div class="bg-white rounded-2xl shadow-sm p-6 flex flex-col items-center text-center">
                <img :src="contact.avatar" class="w-32 h-32 rounded-full object-cover mb-4 border-4 border-slate-50" />
                <h2 class="text-2xl font-bold text-slate-900">{{ contact.name }}</h2>
                <p class="text-slate-500">{{ contact.phone }}</p>
                
                <div class="mt-6 w-full">
                    <button 
                        id="contact-open-chat" 
                        @click="openChat"
                        class="w-full py-3 px-4 bg-blue-600 text-white font-semibold rounded-xl hover:bg-blue-700 shadow-md transition-colors flex items-center justify-center space-x-2"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                            <path fill-rule="evenodd" d="M18 10c0 3.866-3.582 7-8 7a8.841 8.841 0 01-4.083-.98L2 17l1.338-3.123C2.493 12.767 2 11.434 2 10c0-3.866 3.582-7 8-7s8 3.134 8 7zM7 9H5v2h2V9zm8 0h-2v2h2V9zM9 9h2v2H9V9z" clip-rule="evenodd" />
                        </svg>
                        <span>Send Message</span>
                    </button>
                </div>
            </div>

            <!-- Actions -->
            <div class="bg-white rounded-2xl shadow-sm overflow-hidden">
                <button 
                    id="contact-block-button" 
                    @click="goToBlock"
                    class="w-full p-4 flex items-center text-red-600 hover:bg-red-50 transition-colors"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 mr-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
                    </svg>
                    <span class="font-semibold">Block User</span>
                </button>
            </div>
        </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'CONTACT_DETAIL',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const contactId = computed(() => store.selected_contact_id)
    const contact = computed(() => dataStore.contacts.find(c => c.id === contactId.value) || { name: 'Unknown', avatar: '/images/photo1765611339.jpg' })

    const goBackList = async () => {
        store.currentPageId = 'CONTACTS_LIST'
        await router.push({ name: 'CONTACTS_LIST' })
    }

    const openChat = async () => {
        // Find existing chat or create temporary binding
        const existingChat = dataStore.chats.find(c => c.contact_id === contactId.value)
        if (existingChat) {
            store.selected_chat_id = existingChat.id
        } else {
            // In mock app, we might need to handle this gracefully
             store.selected_chat_id = `temp_chat_${contactId.value}`
        }
        store.currentPageId = 'CHAT_THREAD'
        await router.push({ name: 'CHAT_THREAD' })
    }

    const goToBlock = async () => {
        store.currentPageId = 'BLOCK_USER_CONFIRM'
        await router.push({ name: 'BLOCK_USER_CONFIRM' })
    }

    return {
        contact,
        goBackList,
        openChat,
        goToBlock
    }
  }
}
</script>