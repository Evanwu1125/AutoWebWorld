<template>
  <div class="h-screen flex flex-col bg-white">
     <!-- Header -->
    <div class="h-14 border-b flex items-center px-4">
      <button id="back-channel-detail" @click="handleBack" class="mr-4 text-gray-500 hover:text-gray-900">
        ← Cancel
      </button>
      <h2 class="font-bold">Compose Message</h2>
    </div>

    <!-- Compose Area -->
    <div class="flex-1 p-4">
      <textarea 
        id="compose-textarea"
        v-model="inputText"
        @input="handleType"
        class="w-full h-40 border border-gray-300 rounded-md p-3 focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
        placeholder="Type your message..."
      ></textarea>

      <!-- Mention & Emoji Tools -->
      <div class="flex items-center space-x-4 mt-4">
        <div class="relative">
             <button id="mention-button" class="text-gray-500 hover:text-blue-600 font-bold p-2 bg-gray-100 rounded">
                @ Mention
             </button>
             <!-- Hidden input for FSM simulation of typing mention -->
             <input id="mention-input" type="text" class="hidden" @input="handleMention" />
        </div>

        <div class="relative">
             <button id="emoji-toggle" @click="toggleEmoji" class="text-gray-500 hover:text-yellow-500 text-xl p-1">
                ☺
             </button>
             <div v-if="showEmoji" id="emoji-picker" class="absolute top-10 left-0 bg-white shadow-xl border rounded p-2 z-50">
                <span class="emoji-smile cursor-pointer text-2xl hover:bg-gray-100 p-1 rounded" @click="handleEmoji">😊</span>
             </div>
        </div>
      </div>

      <!-- Send Button -->
      <div class="mt-8 flex justify-end">
        <button 
            id="send-message-button" 
            @click="handleSend"
            class="bg-[#007a5a] text-white font-bold py-2 px-6 rounded hover:bg-[#148567] transition disabled:opacity-50 disabled:cursor-not-allowed"
            :disabled="!inputText"
        >
            Send Message
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'MESSAGE_COMPOSE',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const inputText = ref('')
    const showEmoji = ref(false)

    function handleType(e) {
        signatureStore.compose_text = e.target.value
    }

    function handleMention(e) {
        // Mocking the mention input interaction
        signatureStore.compose_has_mention = true
    }

    function toggleEmoji() {
        showEmoji.value = !showEmoji.value
    }

    function handleEmoji() {
        signatureStore.compose_has_emoji = true
        inputText.value += ' 😊'
        signatureStore.compose_text = inputText.value
        showEmoji.value = false
    }

    async function handleSend() {
        if (inputText.value) {
            signatureStore.currentPageId = 'SEND_MESSAGE_SUCCESS'
            await router.push({ name: 'SEND_MESSAGE_SUCCESS' })
        }
    }

    async function handleBack() {
        signatureStore.currentPageId = 'CHANNEL_DETAIL'
        await router.push({ name: 'CHANNEL_DETAIL', params: { id: signatureStore.selected_channel_id } })
    }

    return {
        signatureStore,
        inputText,
        showEmoji,
        handleType,
        handleMention,
        toggleEmoji,
        handleEmoji,
        handleSend,
        handleBack
    }
  }
}
</script>