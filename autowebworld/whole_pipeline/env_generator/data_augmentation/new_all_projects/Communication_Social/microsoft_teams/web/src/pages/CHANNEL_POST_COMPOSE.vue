<template>
  <div class="h-screen flex flex-col bg-gray-50">
    <!-- Header -->
    <header class="bg-[#6264A7] text-white p-4 shadow-md flex justify-between items-center z-20">
      <div class="font-bold text-lg flex items-center">
        <button id="back-to-channels" @click="goBack" class="mr-4 hover:bg-[#464775] p-1 rounded">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
          </svg>
        </button>
        Compose Post <span v-if="currentChannel" class="ml-2 font-normal opacity-80">in #{{ currentChannel.name }}</span>
      </div>
    </header>

    <main class="flex-1 flex flex-col p-6 max-w-4xl mx-auto w-full">
      <div class="bg-white rounded-lg shadow-md p-6 border border-gray-200">
        <h2 class="text-xl font-bold text-gray-800 mb-6">Start a new conversation</h2>
        
        <div class="space-y-6">
          <!-- Subject Input ACT_CHANNEL_POST_TYPE_SUBJECT -->
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Subject</label>
            <input 
              id="channel-post-subject-input"
              type="text" 
              v-model="subject"
              placeholder="Add a subject"
              class="w-full rounded-md border-gray-300 shadow-sm focus:border-[#6264A7] focus:ring-[#6264A7] px-4 py-2 border"
            />
          </div>

          <!-- Body Input ACT_CHANNEL_POST_TYPE_BODY -->
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Message</label>
            <textarea 
              id="channel-post-body-input"
              v-model="body"
              rows="6"
              placeholder="Type your message here..."
              class="w-full rounded-md border-gray-300 shadow-sm focus:border-[#6264A7] focus:ring-[#6264A7] px-4 py-2 border resize-none"
            ></textarea>
          </div>

          <div class="flex justify-end pt-4 border-t border-gray-100">
            <!-- Send Button ACT_CHANNEL_POST_SEND -->
            <button 
              id="channel-post-send-button"
              @click="sendPost"
              :disabled="!isValid"
              class="bg-[#6264A7] hover:bg-[#464775] text-white font-semibold py-2 px-6 rounded-md shadow-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              <span>Send</span>
              <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 transform rotate-45" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { ref, computed, watch } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'CHANNEL_POST_COMPOSE',
  setup() {
    const router = useRouter()
    const route = useRoute()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const channelId = route.params.channelId
    const currentChannel = computed(() => dataStore.channels.find(c => c.id === channelId))

    const subject = ref('')
    const body = ref('')

    const isValid = computed(() => {
      return subject.value.trim().length > 0 && body.value.trim().length > 0
    })

    // Watch for changes and sync to store
    watch(subject, (val) => {
      store.post_subject = val
    })

    watch(body, (val) => {
      store.post_body = val
    })

    const sendPost = async () => {
      if (!isValid.value) return;

      // Update signature store as per effects/actions
      store.post_subject = subject.value;
      store.post_body = body.value;

      store.currentPageId = 'CHANNEL_POST_SENT_SUCCESS';
      await router.push({
        name: 'CHANNEL_POST_SENT_SUCCESS',
        params: { teamId: route.params.teamId, channelId: route.params.channelId }
      });
    }

    const goBack = async () => {
      store.currentPageId = 'CHANNELS_LIST';
      await router.push({
        name: 'CHANNELS_LIST',
        params: { teamId: route.params.teamId }
      });
    }

    return {
      currentChannel,
      subject,
      body,
      isValid,
      sendPost,
      goBack
    }
  }
}
</script>