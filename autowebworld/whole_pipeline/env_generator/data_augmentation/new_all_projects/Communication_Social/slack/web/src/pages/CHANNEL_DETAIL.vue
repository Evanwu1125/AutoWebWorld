<template>
  <div class="h-screen flex flex-col bg-white">
    <!-- Header -->
    <div class="h-14 border-b border-gray-200 flex items-center justify-between px-4">
      <div class="flex items-center">
        <button id="back-channel-list" @click="handleBackList" class="mr-4 text-gray-500 hover:text-gray-900 md:hidden">
          ← Back
        </button>
        <div>
          <h2 class="font-bold text-gray-900 flex items-center">
            <span class="text-gray-400 mr-1">#</span> 
            {{ currentChannel?.name || 'Loading...' }}
          </h2>
          <div class="text-xs text-gray-500">{{ currentChannel?.description || 'No description' }}</div>
        </div>
      </div>
      <div class="flex items-center space-x-2">
         <!-- Settings Button -->
        <button id="channel-header-settings" @click="handleOpenSettings" class="text-gray-500 hover:text-gray-900 p-2 rounded-full hover:bg-gray-100">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
        </button>
      </div>
    </div>

    <!-- Messages Area -->
    <div class="flex-1 overflow-y-auto p-4 custom-scrollbar">
      <div v-for="msg in messages" :key="msg.id" class="mb-4 hover:bg-gray-50 -mx-4 px-4 py-1 group">
        <div class="flex items-start">
          <div class="w-9 h-9 rounded bg-gray-300 mr-3 flex-shrink-0">
             <!-- Placeholder avatar logic -->
          </div>
          <div class="flex-1">
            <div class="flex items-baseline">
              <span class="font-bold text-gray-900 mr-2">{{ getUserName(msg.sender_id) }}</span>
              <span class="text-xs text-gray-500">{{ msg.time }}</span>
            </div>
            <p class="text-gray-800">{{ msg.text }}</p>
            <div v-if="msg.reactions && msg.reactions.length" class="flex mt-1 space-x-1">
              <span v-for="(reaction, idx) in msg.reactions" :key="idx" class="bg-gray-100 rounded-full px-2 py-0.5 text-xs border border-transparent hover:border-gray-300 hover:bg-white cursor-pointer transition">
                {{ reaction }} <span class="text-xs ml-1">1</span>
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Input Area -->
    <div class="p-4 border-t border-gray-200">
      <div class="border border-gray-300 rounded-lg overflow-hidden shadow-sm hover:border-gray-400 transition-colors">
        <!-- Message Input Trigger -->
        <div 
          id="message-input" 
          @click="handleCompose"
          class="bg-white p-3 min-h-[44px] cursor-text text-gray-500"
        >
          Message #{{ currentChannel?.name || 'channel' }}
        </div>
        
        <div class="bg-gray-50 px-2 py-1 flex justify-between items-center border-t border-gray-200">
           <div class="flex space-x-2">
             <button class="p-1 hover:bg-gray-200 rounded text-gray-500 font-bold">B</button>
             <button class="p-1 hover:bg-gray-200 rounded text-gray-500 italic">I</button>
             <button class="p-1 hover:bg-gray-200 rounded text-gray-500 line-through">S</button>
           </div>
           <div class="flex space-x-2">
             <button id="schedule-message-button" @click.stop="handleSchedule" class="p-1 hover:bg-gray-200 rounded text-gray-500" title="Schedule for later">
               <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
             </button>
             <button class="bg-[#007a5a] text-white p-1 rounded hover:bg-[#148567]">
                <svg class="w-4 h-4 transform rotate-90" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path></svg>
             </button>
           </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'CHANNEL_DETAIL',
  setup() {
    const router = useRouter()
    const route = useRoute()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    // Ensure store knows current ID if navigated directly or refreshed
    if (route.params.id) {
        signatureStore.selected_channel_id = route.params.id
    }

    const currentChannel = computed(() => {
      return dataStore.channels.find(c => c.id === signatureStore.selected_channel_id)
    })

    const messages = computed(() => dataStore.messages)

    function getUserName(id) {
        const user = dataStore.users.find(u => u.id === id)
        return user ? user.name : 'Unknown User'
    }

    async function handleCompose() {
      signatureStore.currentPageId = 'MESSAGE_COMPOSE'
      await router.push({ name: 'MESSAGE_COMPOSE' })
    }

    async function handleOpenSettings() {
      signatureStore.currentPageId = 'CHANNEL_SETTINGS'
      await router.push({ name: 'CHANNEL_SETTINGS', params: { id: signatureStore.selected_channel_id } })
    }

    async function handleSchedule() {
      signatureStore.currentPageId = 'MESSAGE_SCHEDULE'
      await router.push({ name: 'MESSAGE_SCHEDULE' })
    }

    async function handleBackList() {
      signatureStore.currentPageId = 'CHANNEL_LIST'
      await router.push({ name: 'CHANNEL_LIST' })
    }

    return {
      signatureStore,
      currentChannel,
      messages,
      getUserName,
      handleCompose,
      handleOpenSettings,
      handleSchedule,
      handleBackList
    }
  }
}
</script>