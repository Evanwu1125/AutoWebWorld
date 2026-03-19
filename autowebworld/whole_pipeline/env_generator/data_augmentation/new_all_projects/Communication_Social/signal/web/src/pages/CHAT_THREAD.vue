<template>
  <div class="h-screen bg-slate-50 flex flex-col">
    <!-- Header -->
    <header class="bg-white shadow-sm z-20 flex items-center justify-between px-4 py-3">
        <div class="flex items-center space-x-3">
            <button id="back-chats-list" @click="goBackChats" class="p-2 text-slate-500 hover:text-blue-600">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
                </svg>
            </button>
            <div class="flex items-center space-x-3 cursor-pointer" id="chat-info-button" @click="goToChatInfo">
                <img :src="contact.avatar" class="w-10 h-10 rounded-full object-cover" />
                <h2 class="font-bold text-slate-800">{{ contact.name }}</h2>
            </div>
        </div>
    </header>

    <!-- Message Area -->
    <div class="flex-1 overflow-y-auto p-4 space-y-4 bg-slate-100">
        <div 
            v-for="msg in messages" 
            :key="msg.id"
            :class="['flex', msg.sender === 'me' ? 'justify-end' : 'justify-start']"
        >
            <div 
                :class="['max-w-xs md:max-w-md px-4 py-2 rounded-2xl shadow-sm', 
                          msg.sender === 'me' ? 'bg-blue-600 text-white rounded-br-none' : 'bg-white text-slate-800 rounded-bl-none']"
            >
                <p>{{ msg.text }}</p>
                <div 
                    :class="['text-xs mt-1 text-right', msg.sender === 'me' ? 'text-blue-100' : 'text-slate-400']"
                >
                    {{ msg.time }}
                </div>
            </div>
        </div>
    </div>

    <!-- Input Area -->
    <div class="bg-white p-4 border-t border-slate-200">
        <div class="flex items-center space-x-2 max-w-2xl mx-auto">
            <input 
                id="message-input"
                type="text" 
                v-model="inputText"
                placeholder="Signal message..."
                class="flex-1 bg-slate-100 rounded-full px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500"
                @keyup.enter="handleSend"
            />
            <button 
                id="send-message-button" 
                @click="goToConfirm"
                :disabled="!inputText.trim()"
                class="p-3 bg-blue-600 text-white rounded-full hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
                <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                    <path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z" />
                </svg>
            </button>
        </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'CHAT_THREAD',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const inputText = ref('')

    const chatId = computed(() => store.selected_chat_id)
    const chat = computed(() => dataStore.chats.find(c => c.id === chatId.value))
    const contact = computed(() => {
        if (!chat.value) return { name: 'Unknown', avatar: '/images/photo1765611271.jpg' }
        return dataStore.contacts.find(c => c.id === chat.value.contact_id) || { name: 'Unknown' }
    })
    
    const messages = computed(() => {
        return dataStore.messages[chatId.value] || []
    })

    const goBackChats = async () => {
        store.currentPageId = 'CHATS_LIST'
        await router.push({ name: 'CHATS_LIST' })
    }

    const goToChatInfo = async () => {
        store.currentPageId = 'CHAT_INFO'
        await router.push({ name: 'CHAT_INFO' })
    }

    const goToConfirm = async () => {
        if (!inputText.value.trim()) return
        store.draft_message_text = inputText.value
        store.currentPageId = 'SEND_MESSAGE_CONFIRM'
        await router.push({ name: 'SEND_MESSAGE_CONFIRM' })
    }
    
    // Helper to allow enter key to trigger confirm
    const handleSend = () => {
        if(inputText.value.trim()) goToConfirm()
    }

    return {
        inputText,
        contact,
        messages,
        goBackChats,
        goToChatInfo,
        goToConfirm,
        handleSend
    }
  }
}
</script>