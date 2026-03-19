<template>
  <div class="flex h-screen bg-black text-white font-sans overflow-hidden">
    <aside class="w-64 bg-black flex-shrink-0 p-6 border-r border-[#282828] hidden md:block">
      <div id="back-track" @click="handleBackTrack" class="flex items-center space-x-2 text-[#B3B3B3] hover:text-white cursor-pointer font-bold mb-8">
         <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M12.5 3.247a1 1 0 0 0-1 0L4 8.75v9a1 1 0 0 0 1 1h5v-5h4v5h5a1 1 0 0 0 1-1v-9l-7.5-5.503z"/></svg>
         <span>Back to Song</span>
      </div>
    </aside>

    <main class="flex-1 flex flex-col relative bg-gradient-to-b from-[#7d4b32] to-[#121212] overflow-hidden">
      <!-- Album Header -->
      <header class="p-8 flex items-end space-x-6 bg-gradient-to-b from-transparent to-black/20">
         <div class="w-52 h-52 shadow-2xl relative group">
           <img :src="album?.image" class="w-full h-full object-cover shadow-lg" />
         </div>
         <div class="flex-1">
            <div class="text-xs font-bold uppercase tracking-widest mb-2">Album</div>
            <h1 class="text-7xl font-bold mb-6 tracking-tight">{{ album?.name }}</h1>
            <div class="text-[#e0e0e0] font-medium text-sm flex items-center space-x-1">
               <div class="flex items-center">
                  <img src="/images/photo1766302470.jpg" class="w-6 h-6 rounded-full mr-2" />
                  <span class="font-bold text-white hover:underline cursor-pointer">The Midnight</span>
               </div>
               <span>•</span>
               <span class="text-[#B3B3B3]">{{ album?.year }}</span>
               <span>•</span>
               <span class="text-[#B3B3B3]">12 songs, 54 min</span>
            </div>
         </div>
      </header>

      <!-- Action Bar -->
      <div class="px-8 py-6 bg-black/20 backdrop-blur-sm sticky top-0 z-20 flex items-center justify-between">
         <div class="flex items-center space-x-6">
            <!-- Play Button -->
            <button class="w-14 h-14 bg-[#1DB954] rounded-full flex items-center justify-center hover:scale-105 hover:bg-[#1ed760] transition-transform shadow-lg">
               <svg class="w-7 h-7 text-black fill-current ml-1" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
            </button>
            
            <!-- Download Toggle -->
            <div 
              id="album-download-toggle" 
              @click="handleDownloadToggle" 
              class="w-8 h-8 border-2 border-[#B3B3B3] rounded-full flex items-center justify-center cursor-pointer hover:border-white hover:text-white text-[#B3B3B3] transition-colors"
              :class="{'bg-[#1DB954] border-[#1DB954] text-black hover:border-[#1ed760] hover:bg-[#1ed760]': isDownloadReady}"
            >
               <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 14l-7 7m0 0l-7-7m7 7V3"/></svg>
            </div>

            <div class="text-[#B3B3B3] hover:text-white cursor-pointer">
               <svg class="w-8 h-8" fill="currentColor" viewBox="0 0 24 24"><path d="M12 8c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm0 2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0 6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z"/></svg>
            </div>
         </div>
      </div>
      
      <!-- Conditional Confirm Link -->
      <div v-if="isDownloadReady" class="px-8 py-2 bg-[#1DB954]/20 border-l-4 border-[#1DB954] mb-4 mx-8 rounded-r flex items-center justify-between">
         <span class="text-sm font-bold">Album ready for download</span>
         <span 
           id="album-download-confirm-link" 
           @click="handleGoToDownloadConfirm" 
           class="text-[#1DB954] font-bold cursor-pointer hover:underline text-sm uppercase tracking-wider"
         >
           Confirm Download
         </span>
      </div>

      <!-- Tracks List -->
      <div class="flex-1 overflow-y-auto px-8 pb-32">
        <table class="w-full text-left border-collapse">
           <thead class="text-[#B3B3B3] text-sm uppercase border-b border-[#282828]">
              <tr>
                 <th class="pb-2 font-medium w-12">#</th>
                 <th class="pb-2 font-medium">Title</th>
                 <th class="pb-2 font-medium text-right w-16"><svg class="w-4 h-4 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/></svg></th>
              </tr>
           </thead>
           <tbody>
              <tr 
                v-for="(track, idx) in tracks" 
                :key="track.id"
                class="group hover:bg-[#ffffff1a] rounded-md transition-colors cursor-pointer"
              >
                 <td class="py-3 text-[#B3B3B3] group-hover:text-white w-12 text-center">
                    <span class="group-hover:hidden">{{ idx + 1 }}</span>
                    <svg class="w-4 h-4 hidden group-hover:inline fill-current" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
                 </td>
                 <td class="py-3">
                    <div class="text-white font-medium group-hover:text-white">{{ track.name }}</div>
                    <div class="text-[#B3B3B3] text-sm group-hover:text-white">The Midnight</div>
                 </td>
                 <td class="py-3 text-[#B3B3B3] text-right group-hover:text-white">{{ track.duration }}</td>
              </tr>
           </tbody>
        </table>
      </div>
    </main>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'
import { useRouter, useRoute } from 'vue-router'

export default {
  name: 'ALBUM_DETAIL',
  setup() {
    const store = useSignatureStore()
    const dataStore = useDataStore()
    const router = useRouter()
    const route = useRoute()

    const albumId = route.params.id || store.selected_album_id
    const album = computed(() => dataStore.albums.find(a => a.id === albumId))
    // Mock tracks for this album
    const tracks = computed(() => dataStore.tracks.filter(t => t.id === 'tr_1' || t.id === 'tr_2'))
    
    const isDownloadReady = computed(() => store.album_download_ready === true)

    const handleBackTrack = async () => {
      // Logic assumes coming from track
      store.setCurrentPageId('TRACK_DETAIL')
      await router.push({ name: 'TRACK_DETAIL' })
    }

    const handleDownloadToggle = () => {
      store.album_download_ready = true
    }

    const handleGoToDownloadConfirm = async () => {
      if (isDownloadReady.value) {
        store.setCurrentPageId('ALBUM_DOWNLOAD_CONFIRM')
        await router.push({ name: 'ALBUM_DOWNLOAD_CONFIRM' })
      }
    }

    return {
      album,
      tracks,
      isDownloadReady,
      handleBackTrack,
      handleDownloadToggle,
      handleGoToDownloadConfirm
    }
  }
}
</script>