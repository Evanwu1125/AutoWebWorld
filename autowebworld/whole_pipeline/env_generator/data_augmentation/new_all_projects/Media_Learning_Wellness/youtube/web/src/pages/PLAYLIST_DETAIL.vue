<template>
  <div class="min-h-screen bg-[#0F0F0F] text-white flex flex-col">
    <!-- Navbar -->
    <nav class="sticky top-0 z-50 bg-[#0F0F0F]/95 backdrop-blur border-b border-gray-800 px-4 h-14 flex items-center justify-between">
      <div class="flex items-center gap-4">
        <button 
          id="playlist-back-library"
          @click="goBackLibrary"
          class="flex items-center gap-2 hover:text-gray-300 transition-colors text-sm font-medium"
        >
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>
          Back to Library
        </button>
      </div>
      <div class="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center text-sm font-bold">U</div>
    </nav>

    <main class="flex-1 max-w-7xl mx-auto w-full p-6 flex flex-col md:flex-row gap-8">
      <!-- Left Column: Playlist Info -->
      <div class="md:w-80 flex-shrink-0">
        <div class="bg-gradient-to-b from-[#333] to-[#1F1F1F] p-6 rounded-2xl h-fit sticky top-20 border border-gray-700">
           <div class="aspect-video w-full bg-gray-800 rounded-xl mb-6 overflow-hidden shadow-lg">
              <img :src="playlist?.image" :alt="playlist?.title" class="w-full h-full object-cover opacity-80">
           </div>
           
           <h1 class="text-2xl font-bold mb-2">{{ playlist?.title }}</h1>
           <div class="text-sm text-gray-400 mb-6">
             <p>Updated today</p>
             <p>{{ playlist?.count }} videos</p>
             <p>Private • User Name</p>
           </div>
           
           <div class="flex flex-col gap-3">
             <button 
               id="playlist-first-video" 
               @click="goWatchFirst"
               class="w-full bg-white text-black font-bold py-3 rounded-full hover:bg-gray-200 transition-colors flex items-center justify-center gap-2"
             >
               <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
               Play All
             </button>
             <button class="w-full bg-[#333] text-white font-bold py-3 rounded-full hover:bg-[#444] transition-colors flex items-center justify-center gap-2">
               <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path></svg>
               Shuffle
             </button>
           </div>
        </div>
      </div>

      <!-- Right Column: Video List -->
      <div class="flex-1">
        <div class="space-y-2">
          <div v-for="i in 10" :key="i" class="flex gap-4 p-3 hover:bg-[#1F1F1F] rounded-xl cursor-pointer group items-center">
            <span class="text-gray-500 w-6 text-center">{{ i }}</span>
            <div class="w-40 aspect-video bg-gray-800 rounded-lg overflow-hidden relative">
               <div class="absolute bottom-1 right-1 bg-black/80 text-white text-[10px] px-1 rounded">5:20</div>
            </div>
            <div class="flex-1 min-w-0">
               <h3 class="font-bold text-sm mb-1 truncate text-gray-200 group-hover:text-white">Video Title in Playlist {{ i }}</h3>
               <div class="text-xs text-gray-400">Channel Name • 1.5M views</div>
            </div>
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
  name: 'PLAYLIST_DETAIL',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const playlist = computed(() => {
      return dataStore.playlists.find(p => p.id === store.selected_playlist_id) || dataStore.playlists[0]
    })

    const goBackLibrary = () => {
      store.currentPageId = 'LIBRARY'
      router.push({ name: 'LIBRARY' })
    }

    const goWatchFirst = () => {
      store.currentPageId = 'WATCH_VIDEO'
      router.push({ name: 'WATCH_VIDEO' })
    }

    return {
      playlist,
      goBackLibrary,
      goWatchFirst
    }
  }
}
</script>