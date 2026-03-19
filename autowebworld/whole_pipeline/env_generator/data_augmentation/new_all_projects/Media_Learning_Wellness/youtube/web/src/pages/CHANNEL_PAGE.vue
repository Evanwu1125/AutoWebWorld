<template>
  <div class="min-h-screen bg-[#0F0F0F] text-white flex flex-col">
    <!-- Navbar -->
    <nav class="sticky top-0 z-50 bg-[#0F0F0F]/95 backdrop-blur border-b border-gray-800 px-4 h-14 flex items-center justify-between">
      <div class="flex items-center gap-4">
        <button 
          id="channel-back-subscriptions"
          @click="goBackSubscriptions"
          class="p-2 hover:bg-gray-800 rounded-full"
        >
          <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>
        </button>
        <div class="flex items-center gap-1">
          <div class="bg-red-600 text-white rounded-lg p-1">
            <svg class="w-6 h-6 fill-current" viewBox="0 0 24 24"><path d="M19.615 3.184c-3.604-.246-11.631-.245-15.23 0-3.897.266-4.356 2.62-4.385 8.816.029 6.185.484 8.549 4.385 8.816 3.6.245 11.626.246 15.23 0 3.897-.266 4.356-2.62 4.385-8.816-.029-6.185-.484-8.549-4.385-8.816zm-10.615 12.816v-8l8 3.993-8 4.007z"/></svg>
          </div>
        </div>
      </div>
      <div class="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center text-sm font-bold">U</div>
    </nav>

    <main class="flex-1 max-w-7xl mx-auto w-full">
      <!-- Channel Banner -->
      <div class="h-48 md:h-64 w-full bg-gradient-to-r from-gray-800 to-gray-900 overflow-hidden relative">
         <!-- Simulated banner image -->
         <div class="absolute inset-0 bg-blue-900/30"></div>
      </div>

      <!-- Channel Header -->
      <div class="px-6 py-6 border-b border-gray-800">
        <div class="flex flex-col md:flex-row items-start md:items-center gap-6">
          <div class="w-32 h-32 rounded-full overflow-hidden border-4 border-[#0F0F0F] -mt-16 bg-[#272727] relative z-10">
            <img :src="channel?.avatar" :alt="channel?.name" class="w-full h-full object-cover">
          </div>
          
          <div class="flex-1">
            <h1 class="text-3xl font-bold mb-1">{{ channel?.name }}</h1>
            <div class="text-gray-400 text-sm mb-4">
              {{ channel?.subscribers }} subscribers • 432 videos
            </div>
            
            <button 
              id="channel-subscribe-button" 
              @click="goSubscribeConfirm"
              class="bg-white text-black px-8 py-2.5 rounded-full font-bold hover:bg-gray-200 transition-colors"
            >
              Subscribe
            </button>
          </div>
        </div>

        <!-- Channel Navigation Tabs -->
        <div class="flex gap-8 mt-8 text-sm font-medium border-b border-transparent">
          <div class="border-b-2 border-white pb-3 cursor-pointer">HOME</div>
          <div class="text-gray-400 hover:text-white pb-3 cursor-pointer transition-colors">VIDEOS</div>
          <div class="text-gray-400 hover:text-white pb-3 cursor-pointer transition-colors">PLAYLISTS</div>
          <div class="text-gray-400 hover:text-white pb-3 cursor-pointer transition-colors">COMMUNITY</div>
        </div>
      </div>

      <!-- Featured Content -->
      <div class="p-6">
        <h2 class="text-lg font-bold mb-4">For You</h2>
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div v-for="i in 4" :key="i" class="cursor-pointer group">
            <div class="aspect-video bg-gray-800 rounded-xl mb-2 overflow-hidden">
               <div class="w-full h-full bg-gray-700/50"></div>
            </div>
            <h3 class="font-bold text-sm line-clamp-2 group-hover:text-blue-400">Sample Video Title for {{ channel?.name }}</h3>
            <div class="text-xs text-gray-400 mt-1">10K views • 1 day ago</div>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'CHANNEL_PAGE',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const channel = computed(() => {
      return dataStore.channels.find(c => c.id === store.selected_channel_id) || dataStore.channels[0]
    })

    const goBackSubscriptions = () => {
      store.currentPageId = 'SUBSCRIPTIONS'
      router.push({ name: 'SUBSCRIPTIONS' })
    }

    const goSubscribeConfirm = () => {
      store.currentPageId = 'CHANNEL_SUBSCRIBE_CONFIRM'
      router.push({ name: 'CHANNEL_SUBSCRIBE_CONFIRM' })
    }

    return {
      channel,
      goBackSubscriptions,
      goSubscribeConfirm
    }
  }
}
</script>