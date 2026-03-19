<template>
  <div class="h-screen flex flex-col bg-white">
     <!-- Header -->
    <div class="h-14 border-b flex items-center px-4">
      <button id="back-dm-detail" @click="handleBack" class="mr-4 text-gray-500 hover:text-gray-900">
        ← Cancel
      </button>
      <h2 class="font-bold">Direct Message</h2>
    </div>

    <div class="flex-1 p-4 flex flex-col">
       <div class="flex-1"></div> <!-- Spacer to push input to bottom like mobile or chat app -->
       
       <div class="border-t pt-4">
           <textarea 
            id="dm-compose-textarea"
            v-model="text"
            @input="handleType"
            class="w-full h-20 border border-gray-300 rounded-md p-3 resize-none focus:ring-2 focus:ring-blue-500"
            placeholder="Type your message..."
          ></textarea>
          
          <div class="mt-2 flex justify-end">
              <button 
                id="dm-send-message-button"
                @click="handleSend"
                class="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:opacity-50"
                :disabled="!text"
              >
                Send
              </button>
          </div>
       </div>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'DM_COMPOSE',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const text = ref('')

    function handleType(e) {
        signatureStore.dm_compose_text = e.target.value
    }

    async function handleSend() {
        if (text.value) {
            signatureStore.currentPageId = 'START_DM_SUCCESS'
            await router.push({ name: 'START_DM_SUCCESS' })
        }
    }

    async function handleBack() {
        signatureStore.currentPageId = 'DM_DETAIL'
        await router.push({ name: 'DM_DETAIL', params: { id: signatureStore.selected_dm_id } })
    }

    return {
        signatureStore,
        text,
        handleType,
        handleSend,
        handleBack
    }
  }
}
</script>