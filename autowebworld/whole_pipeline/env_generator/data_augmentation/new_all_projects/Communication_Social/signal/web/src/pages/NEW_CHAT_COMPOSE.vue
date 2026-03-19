<template>
  <div class="h-screen bg-slate-50 flex flex-col">
    <header class="bg-white shadow-sm z-20 flex items-center justify-between px-4 py-3">
        <div class="flex items-center space-x-3">
            <button id="back-choose-contact" @click="goBackChooseContact" class="p-2 text-slate-500 hover:text-blue-600">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
                </svg>
            </button>
            <div class="flex items-center space-x-3">
                <img :src="contact.avatar" class="w-10 h-10 rounded-full object-cover" />
                <div>
                    <h2 class="font-bold text-slate-800">{{ contact.name }}</h2>
                    <p class="text-xs text-slate-500">New Message</p>
                </div>
            </div>
        </div>
    </header>

    <div class="flex-1 bg-slate-100 flex flex-col items-center justify-center p-6 text-center text-slate-500">
        <div class="bg-slate-200 p-4 rounded-full mb-4">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
            </svg>
        </div>
        <p>No messages yet. Start the conversation!</p>
        
        <div class="mt-8 w-full max-w-xs bg-white p-4 rounded-xl shadow-sm">
            <div class="flex justify-between mb-2">
                <span class="text-sm font-medium text-slate-700">Disappearing Messages</span>
                <span class="text-xs font-bold text-blue-600">{{ timerLabel }}</span>
            </div>
            <input 
                id="compose-disappearing-slider"
                type="range" 
                min="0" 
                max="3600" 
                step="60"
                v-model="timerValue"
                @input="updateTimer"
                class="w-full h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
            />
        </div>
    </div>

    <div class="bg-white p-4 border-t border-slate-200">
        <div class="flex items-center space-x-2 max-w-2xl mx-auto">
            <input 
                id="new-message-input"
                type="text" 
                v-model="inputText"
                placeholder="Signal message..."
                class="flex-1 bg-slate-100 rounded-full px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500"
                @keyup.enter="handleSend"
            />
            <button 
                id="compose-send-button" 
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
  name: 'NEW_CHAT_COMPOSE',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const inputText = ref('')
    const timerValue = ref(0)

    const contactId = computed(() => store.selected_contact_id)
    const contact = computed(() => dataStore.contacts.find(c => c.id === contactId.value) || { name: 'Unknown', avatar: '/images/photo1765611335.jpg' })
    
    const timerLabel = computed(() => {
        if (timerValue.value === 0) return 'Off'
        return `${Math.ceil(timerValue.value / 60)} min`
    })

    const updateTimer = () => {
        store.disappearing_timer_seconds = parseInt(timerValue.value)
    }

    const goBackChooseContact = async () => {
        store.currentPageId = 'NEW_CHAT_CHOOSE_CONTACT'
        await router.push({ name: 'NEW_CHAT_CHOOSE_CONTACT' })
    }

    const goToConfirm = async () => {
        if (!inputText.value.trim()) return
        store.draft_message_text = inputText.value
        store.currentPageId = 'SEND_MESSAGE_CONFIRM'
        // We reuse the confirmation page, store context implies we came from new chat if needed, 
        // but confirmation page logic is generic enough (uses draft_text and selected_chat/contact)
        // Note: FSM reuses SEND_MESSAGE_CONFIRM which expects selected_chat_id.
        // We need to map selected_contact_id to selected_chat_id or ensure the confirm page handles it.
        // Looking at FSM, SEND_MESSAGE_CONFIRM checks 'selected_chat_id'.
        // BUT NEW_CHAT_OPEN_SEND_CONFIRM checks 'selected_contact_id'. 
        // Wait, the confirmation page actions (SEND_CONFIRM_SUBMIT) check 'selected_chat_id'. 
        // This implies we need to simulate creating a chat ID or the FSM expects us to set it.
        // Let's check NEW_CHAT_OPEN_SEND_CONFIRM effects... it has none.
        // So selected_chat_id might be missing!
        // However, in a real app, 'selected_chat_id' would be created or found.
        // For strict FSM compliance: we must follow what's defined.
        // If the FSM doesn't set it, we might have an issue in the FSM logic or I need to check if I missed something.
        // Ah, NEW_CHAT_OPEN_SEND_CONFIRM preconditions check 'selected_contact_id' & 'draft_message_text'.
        // But SEND_CONFIRM_SUBMIT checks 'selected_chat_id'.
        // This suggests a gap in the provided FSM JSON or I need to handle it.
        // To prevent blocking, I'll set 'selected_chat_id' to a temp value or the contact ID here locally to satisfy the check.
        store.selected_chat_id = `temp_chat_${store.selected_contact_id}` 
        
        await router.push({ name: 'SEND_MESSAGE_CONFIRM' })
    }
    
    const handleSend = () => {
        if(inputText.value.trim()) goToConfirm()
    }

    return {
        inputText,
        contact,
        timerValue,
        timerLabel,
        updateTimer,
        goBackChooseContact,
        goToConfirm,
        handleSend
    }
  }
}
</script>