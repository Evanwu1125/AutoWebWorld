<template>
  <div class="h-screen bg-slate-50 flex flex-col">
    <header class="bg-white shadow-sm z-20 px-4 py-3 flex items-center">
        <button id="chat-info-back-thread" @click="goBackThread" class="p-2 text-slate-500 hover:text-blue-600 mr-4">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
            </svg>
        </button>
        <h1 class="text-xl font-bold text-slate-800">Chat Info</h1>
    </header>

    <div class="flex-1 overflow-y-auto p-4">
        <div class="max-w-md mx-auto space-y-6">
            <div class="bg-white rounded-2xl shadow-sm p-6 flex flex-col items-center text-center">
                <img :src="contact.avatar" class="w-32 h-32 rounded-full object-cover mb-4 border-4 border-slate-50" />
                <h2 class="text-2xl font-bold text-slate-900">{{ contact.name }}</h2>
            </div>

            <div class="bg-white rounded-2xl shadow-sm overflow-hidden divide-y divide-slate-100">
                <button 
                    id="chat-info-disappearing" 
                    @click="goToDisappearing"
                    class="w-full p-4 flex items-center justify-between hover:bg-slate-50 transition-colors"
                >
                    <div class="flex items-center space-x-3 text-slate-800">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <span class="font-medium">Disappearing Messages</span>
                    </div>
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
                    </svg>
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
  name: 'CHAT_INFO',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const chatId = computed(() => store.selected_chat_id)
    const chat = computed(() => dataStore.chats.find(c => c.id === chatId.value))
    const contact = computed(() => {
        if (!chat.value) return { name: 'Unknown', avatar: '/images/photo1765611494.jpg' }
        return dataStore.contacts.find(c => c.id === chat.value.contact_id) || { name: 'Unknown', avatar: '/images/photo1765611494.jpg' }
    })

    const goBackThread = async () => {
        store.currentPageId = 'CHAT_THREAD'
        await router.push({ name: 'CHAT_THREAD' })
    }

    const goToDisappearing = async () => {
        store.currentPageId = 'DISAPPEARING_MESSAGES_SETTINGS'
        await router.push({ name: 'DISAPPEARING_MESSAGES_SETTINGS' })
    }

    return {
        contact,
        goBackThread,
        goToDisappearing
    }
  }
}
</script>