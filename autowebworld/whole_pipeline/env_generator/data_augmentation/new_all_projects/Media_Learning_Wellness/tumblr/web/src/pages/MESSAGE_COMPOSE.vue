<template>
  <div class="min-h-screen bg-white flex flex-col">
    <!-- Header -->
    <div class="p-4 border-b border-gray-100 flex items-center justify-between">
      <button 
        id="message-compose-back-inbox" 
        @click="goBack"
        class="text-gray-400 hover:text-gray-600 font-bold text-sm"
      >
        Cancel
      </button>
      <h2 class="font-bold text-gray-800">New Message</h2>
      <button 
        id="message-send-button" 
        @click="sendMessage"
        :disabled="!isValid"
        :class="[
          'px-4 py-1.5 rounded-full font-bold text-sm transition-all',
          isValid ? 'bg-blue-500 text-white hover:bg-blue-600' : 'bg-gray-100 text-gray-300 cursor-not-allowed'
        ]"
      >
        Send
      </button>
    </div>

    <!-- Form -->
    <div class="p-4 space-y-4">
      <div class="border-b border-gray-100 pb-2">
         <label class="block text-xs font-bold text-gray-400 uppercase mb-1">To:</label>
         <div 
           id="message-recipient-input"
           @click="focusRecipient"
         >
           <input 
             ref="recipientInput"
             type="text" 
             placeholder="Username" 
             class="w-full outline-none text-lg font-bold text-gray-800 placeholder-gray-300"
             :value="store.message_recipient"
             @input="handleRecipientInput"
           />
         </div>
      </div>

      <div class="flex-1">
         <div 
           id="message-body-textarea" 
           @click="focusBody"
           class="min-h-[200px] cursor-text"
         >
           <textarea 
             ref="bodyInput"
             placeholder="Say something nice..." 
             class="w-full h-full min-h-[300px] resize-none outline-none text-lg text-gray-700 placeholder-gray-300"
             :value="store.message_body"
             @input="handleBodyInput"
           ></textarea>
         </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'MESSAGE_COMPOSE',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const recipientInput = ref(null)
    const bodyInput = ref(null)

    const focusRecipient = () => recipientInput.value?.focus()
    const focusBody = () => bodyInput.value?.focus()

    const handleRecipientInput = (e) => store.message_recipient = e.target.value
    const handleBodyInput = (e) => store.message_body = e.target.value

    const isValid = computed(() => {
      return (store.message_recipient?.length > 0) && (store.message_body?.length > 0)
    })

    const goBack = async () => {
      store.currentPageId = 'MESSAGES_INBOX'
      await router.push({ name: 'MESSAGES_INBOX' })
    }

    const sendMessage = async () => {
      if (!isValid.value) return
      store.success_message = `Message sent to ${store.message_recipient}`
      store.currentPageId = 'MESSAGE_SEND_SUCCESS'
      await router.push({ name: 'MESSAGE_SEND_SUCCESS' })
    }

    return {
      store,
      recipientInput,
      bodyInput,
      focusRecipient,
      focusBody,
      handleRecipientInput,
      handleBodyInput,
      isValid,
      goBack,
      sendMessage
    }
  }
}
</script>