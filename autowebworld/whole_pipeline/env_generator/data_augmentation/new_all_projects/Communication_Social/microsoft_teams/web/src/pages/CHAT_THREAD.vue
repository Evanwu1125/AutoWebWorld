<template>
  <div class="h-screen flex flex-col bg-gray-50">
    <!-- Header -->
    <header class="bg-white text-gray-800 p-4 shadow-sm border-b border-gray-200 flex justify-between items-center z-20">
      <div class="font-bold text-lg flex items-center">
        <button id="chat-thread-back" @click="goBack" class="mr-4 hover:bg-gray-100 p-1 rounded">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
          </svg>
        </button>
        <div class="flex items-center gap-3">
             <img 
                v-if="currentChat"
                :src="currentChat.image" 
                class="w-8 h-8 rounded-full object-cover" 
                alt="Avatar"
                @error="$event.target.src = 'https://picsum.photos/100/100'"
             />
             <span>{{ currentChat?.name || 'Chat' }}</span>
        </div>
      </div>
    </header>

    <!-- Messages Area -->
    <main class="flex-1 overflow-y-auto p-6 space-y-6 bg-white">
        <div v-if="!messages.length" class="text-center text-gray-400 mt-10">No messages yet. Start the conversation!</div>
        
        <div 
            v-for="msg in messages" 
            :key="msg.id" 
            :class="`flex ${msg.sender === 'me' ? 'justify-end' : 'justify-start'}`"
        >
            <div 
                :class="`max-w-[70%] rounded-lg p-3 ${msg.sender === 'me' ? 'bg-[#E8E8FA] text-gray-800' : 'bg-gray-100 text-gray-800'}`"
            >
                <p>{{ msg.text }}</p>
                <div class="text-xs text-gray-400 mt-1 text-right">{{ msg.time }}</div>
            </div>
        </div>
    </main>

    <!-- Input Area -->
    <footer class="p-4 bg-white border-t border-gray-200">
        <div class="flex gap-2">
            <input 
              id="chat-message-input"
              type="text" 
              v-model="inputMessage"
              @keypress.enter="sendMessage"
              placeholder="Type a new message"
              class="flex-1 rounded-md border-gray-300 shadow-sm focus:border-[#6264A7] focus:ring-[#6264A7] px-4 py-2 border"
            />
            <button 
              id="chat-send-button"
              @click="sendMessage"
              :disabled="!isValid"
              class="bg-[#6264A7] hover:bg-[#464775] text-white p-2 rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 transform rotate-90" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
            </button>
        </div>
    </footer>
  </div>
</template>

<script>
import { ref, computed, watch } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'CHAT_THREAD',
  setup() {
    const router = useRouter()
    const route = useRoute()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const chatId = route.params.chatId
    const currentChat = computed(() => dataStore.chats.find(c => c.id === chatId))
    const messages = computed(() => dataStore.messages.filter(m => m.chatId === chatId))

    const inputMessage = ref('')

    const isValid = computed(() => {
      return inputMessage.value.trim().length > 0
    })

    // Watch for changes and sync to store
    watch(inputMessage, (val) => {
      store.new_message_text = val
    })

    const sendMessage = async () => {
      if (!isValid.value) return;

      store.new_message_text = inputMessage.value;
      store.currentPageId = 'CHAT_MESSAGE_SENT_SUCCESS';
      await router.push({ name: 'CHAT_MESSAGE_SENT_SUCCESS', params: { chatId } });
    }

    const goBack = async () => {
      store.currentPageId = 'CHAT_LIST';
      await router.push({ name: 'CHAT_LIST' });
    }

    return {
      currentChat,
      messages,
      inputMessage,
      isValid,
      sendMessage,
      goBack
    }
  }
}
</script>