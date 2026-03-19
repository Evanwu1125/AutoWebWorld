<template>
  <div v-if="message" class="min-h-screen bg-slate-900 text-white flex flex-col h-screen">
    <!-- Header -->
    <header class="bg-slate-800 border-b border-slate-700 p-4 flex items-center gap-4 shadow-md z-10">
      <button 
        id="thread-back-inbox" 
        @click="goBackInbox"
        class="p-2 hover:bg-slate-700 rounded-full transition-colors text-slate-400"
      >
        <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" /></svg>
      </button>
      
      <div class="flex items-center gap-3">
        <img :src="message.avatar" class="w-10 h-10 rounded-full object-cover" />
        <div>
           <h1 class="font-bold">{{ message.recipient_name }}</h1>
           <span class="text-xs text-green-400">Online</span>
        </div>
      </div>
    </header>

    <!-- Chat Area -->
    <div class="flex-1 overflow-y-auto p-4 space-y-4 bg-slate-900 scroll-smooth">
      <div 
        v-for="msg in threadDetails" 
        :key="msg.id"
        :class="['flex', msg.sender === 'me' ? 'justify-end' : 'justify-start']"
      >
        <div 
          :class="[
            'max-w-[70%] rounded-2xl p-3 px-4 text-sm leading-relaxed shadow-sm',
            msg.sender === 'me' ? 'bg-blue-500 text-white rounded-br-none' : 'bg-slate-700 text-slate-200 rounded-bl-none'
          ]"
        >
          {{ msg.text }}
          <div :class="['text-[10px] mt-1 opacity-70', msg.sender === 'me' ? 'text-blue-100' : 'text-slate-400']">
             {{ formatTime(msg.timestamp) }}
          </div>
        </div>
      </div>
    </div>

    <!-- Footer Actions -->
    <div class="p-4 bg-slate-800 border-t border-slate-700">
      <button 
        id="thread-reply-button" 
        @click="goReply"
        class="w-full bg-slate-700 hover:bg-slate-600 text-slate-300 py-3 rounded-full text-left px-6 transition-colors flex justify-between items-center group"
      >
        <span>Reply...</span>
        <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 opacity-0 group-hover:opacity-100 transition-opacity transform group-hover:translate-x-1" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" /></svg>
      </button>
    </div>
  </div>
</template>

<script>
import { computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'MESSAGE_THREAD',
  setup() {
    const route = useRoute()
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const msgId = computed(() => route.params.id || store.selected_message_id)
    const message = computed(() => dataStore.messages.find(m => m.id === msgId.value))
    
    // Get thread details or fallback
    const threadDetails = computed(() => {
       return dataStore.messageThreads[msgId.value] || [
         { id: 'fallback', sender: 'them', text: message.value?.last_message || 'Hello', timestamp: new Date().toISOString() }
       ]
    })

    const formatTime = (isoString) => {
      return new Date(isoString).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }

    const goBackInbox = async () => {
      store.currentPageId = 'MESSAGES_INBOX'
      await router.push({ name: 'MESSAGES_INBOX' })
    }

    const goReply = async () => {
      // Maps to MESSAGE_COMPOSE in FSM, pre-filling recipient would be nice but FSM starts fresh
      store.currentPageId = 'MESSAGE_COMPOSE'
      await router.push({ name: 'MESSAGE_COMPOSE' })
    }

    onMounted(() => {
      if (!msgId.value) router.push({ name: 'MESSAGES_INBOX' })
    })

    return {
      message,
      threadDetails,
      formatTime,
      goBackInbox,
      goReply
    }
  }
}
</script>