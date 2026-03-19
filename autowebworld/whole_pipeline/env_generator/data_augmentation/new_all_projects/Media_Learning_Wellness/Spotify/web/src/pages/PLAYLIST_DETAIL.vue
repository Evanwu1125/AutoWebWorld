<template>
  <div class="flex h-screen bg-black text-white font-sans overflow-hidden">
    <aside class="w-64 bg-black flex-shrink-0 p-6 border-r border-[#282828] hidden md:block">
      <div id="back-library" @click="handleBackLibrary" class="flex items-center space-x-2 text-[#B3B3B3] hover:text-white cursor-pointer font-bold mb-8">
         <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M12.5 3.247a1 1 0 0 0-1 0L4 8.75v9a1 1 0 0 0 1 1h5v-5h4v5h5a1 1 0 0 0 1-1v-9l-7.5-5.503z"/></svg>
         <span>Your Library</span>
      </div>
    </aside>

    <main class="flex-1 flex flex-col relative bg-gradient-to-b from-[#477d95] to-[#121212] overflow-hidden">
      <!-- Playlist Header -->
      <header class="p-8 flex items-end space-x-6 bg-gradient-to-b from-transparent to-black/20">
         <div class="w-52 h-52 shadow-2xl relative group">
           <img :src="playlist?.image" class="w-full h-full object-cover shadow-lg" />
         </div>
         <div class="flex-1">
            <div class="text-xs font-bold uppercase tracking-widest mb-2">Playlist</div>
            <h1 class="text-7xl font-bold mb-6 tracking-tight">{{ playlist?.name }}</h1>
            <div class="text-[#e0e0e0] font-medium text-sm flex items-center space-x-1">
               <span class="font-bold text-white">{{ playlist?.owner }}</span>
               <span>•</span>
               <span class="text-[#B3B3B3]">{{ tracks.length }} songs, 2 hr 15 min</span>
            </div>
            <p class="text-[#B3B3B3] mt-2 opacity-80">{{ playlist?.description }}</p>
         </div>
      </header>

      <!-- Action Bar -->
      <div class="px-8 py-6 bg-black/20 backdrop-blur-sm sticky top-0 z-20 flex items-center justify-between">
         <div class="flex items-center space-x-6">
            <!-- Play Button -->
            <button class="w-14 h-14 bg-[#1DB954] rounded-full flex items-center justify-center hover:scale-105 hover:bg-[#1ed760] transition-transform shadow-lg">
               <svg class="w-7 h-7 text-black fill-current ml-1" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
            </button>
            
            <!-- Save to Library -->
            <button id="playlist-save-to-library" @click="handleAddToLibrary" class="text-[#B3B3B3] hover:text-white transition-colors">
               <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"/></svg>
            </button>

            <!-- More Menu -->
            <div id="playlist-more-menu" class="relative group">
               <button class="text-[#B3B3B3] hover:text-white transition-colors">
                  <svg class="w-8 h-8" fill="currentColor" viewBox="0 0 24 24"><path d="M12 8c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm0 2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0 6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z"/></svg>
               </button>
               <div class="hidden group-hover:block absolute left-0 top-full mt-2 w-48 bg-[#282828] rounded shadow-xl z-50 border border-[#3E3E3E]">
                  <div class="item-share px-4 py-3 hover:bg-[#3E3E3E] cursor-pointer text-sm" @click="handleShare">Share</div>
                  <div class="item-delete px-4 py-3 hover:bg-[#3E3E3E] cursor-pointer text-sm">Delete</div>
               </div>
            </div>
         </div>

         <!-- Search within playlist -->
         <div class="relative">
             <input 
               id="playlist-search-input"
               v-model="searchQuery"
               @keyup.enter="handleSearch"
               class="bg-transparent border-b border-[#B3B3B3] text-white py-1 px-2 focus:outline-none focus:border-white w-48 text-sm"
               placeholder="Search in playlist"
             />
             <svg class="w-4 h-4 text-white absolute right-0 top-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
         </div>
      </div>

      <!-- Tracks List -->
      <!-- Search Results Container -->
      <div
        v-if="hasSearched"
        id="playlist-search-results"
        class="flex-1 overflow-y-auto px-8 pb-32"
      >
        <h2 class="text-xl font-bold mb-4 mt-4">Search Results</h2>
        <table class="w-full text-left border-collapse">
           <thead class="text-[#B3B3B3] text-sm uppercase border-b border-[#282828]">
              <tr>
                 <th class="pb-2 font-medium w-12">#</th>
                 <th class="pb-2 font-medium">Title</th>
                 <th class="pb-2 font-medium hidden md:table-cell">Album</th>
                 <th class="pb-2 font-medium text-right w-16"><svg class="w-4 h-4 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/></svg></th>
              </tr>
           </thead>
           <tbody>
              <tr
                v-for="(track, idx) in searchResults"
                :key="track.id"
                class="track-row-matched group hover:bg-[#ffffff1a] rounded-md transition-colors cursor-pointer"
                :class="`data-id-${track.id}`"
                @click="handleClickMatchedTrack(track)"
              >
                 <td class="py-3 text-[#B3B3B3] group-hover:text-white w-12 text-center">
                    <span class="group-hover:hidden">{{ idx + 1 }}</span>
                    <svg class="w-4 h-4 hidden group-hover:inline fill-current" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
                 </td>
                 <td class="py-3">
                    <div class="flex items-center">
                       <img :src="track.image" class="w-10 h-10 mr-4 rounded" />
                       <div>
                          <div class="text-white font-medium mb-1 group-hover:text-white">{{ track.name }}</div>
                          <div class="text-[#B3B3B3] text-sm group-hover:text-white">{{ track.artist }}</div>
                       </div>
                    </div>
                 </td>
                 <td class="py-3 text-[#B3B3B3] group-hover:text-white hidden md:table-cell">{{ track.album }}</td>
                 <td class="py-3 text-[#B3B3B3] text-right group-hover:text-white">{{ track.duration }}</td>
              </tr>
           </tbody>
        </table>
      </div>

      <!-- All Tracks Container -->
      <div
        v-else
        id="playlist-tracks"
        class="flex-1 overflow-y-auto px-8 pb-32"
      >
        <table class="w-full text-left border-collapse">
           <thead class="text-[#B3B3B3] text-sm uppercase border-b border-[#282828]">
              <tr>
                 <th class="pb-2 font-medium w-12">#</th>
                 <th class="pb-2 font-medium">Title</th>
                 <th class="pb-2 font-medium hidden md:table-cell">Album</th>
                 <th class="pb-2 font-medium text-right w-16"><svg class="w-4 h-4 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/></svg></th>
              </tr>
           </thead>
           <tbody>
              <tr
                v-for="(track, idx) in tracks"
                :key="track.id"
                class="track-row-visible group hover:bg-[#ffffff1a] rounded-md transition-colors cursor-pointer"
                :class="`data-id-${track.id}`"
                @click="handleClickTrack(track)"
              >
                 <td class="py-3 text-[#B3B3B3] group-hover:text-white w-12 text-center">
                    <span class="group-hover:hidden">{{ idx + 1 }}</span>
                    <svg class="w-4 h-4 hidden group-hover:inline fill-current" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
                 </td>
                 <td class="py-3">
                    <div class="flex items-center">
                       <img :src="track.image" class="w-10 h-10 mr-4 rounded" />
                       <div>
                          <div class="text-white font-medium mb-1 group-hover:text-white">{{ track.name }}</div>
                          <div class="text-[#B3B3B3] text-sm group-hover:text-white">{{ track.artist }}</div>
                       </div>
                    </div>
                 </td>
                 <td class="py-3 text-[#B3B3B3] group-hover:text-white hidden md:table-cell">{{ track.album }}</td>
                 <td class="py-3 text-[#B3B3B3] text-right group-hover:text-white">{{ track.duration }}</td>
              </tr>
           </tbody>
        </table>
      </div>
    </main>
  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'
import { useRouter, useRoute } from 'vue-router'

export default {
  name: 'PLAYLIST_DETAIL',
  setup() {
    const store = useSignatureStore()
    const dataStore = useDataStore()
    const router = useRouter()
    const route = useRoute()

    const playlistId = route.params.id || store.selected_playlist_id
    const playlist = computed(() => dataStore.playlists.find(p => p.id === playlistId))
    const tracks = computed(() => dataStore.tracks) // Mock: showing all tracks for demo
    
    const searchQuery = ref('')
    const hasSearched = computed(() => store.playlist_detail_has_searched === true)

    const searchResults = computed(() => {
       if (!searchQuery.value) return []
       return tracks.value.filter(t => t.name.toLowerCase().includes(searchQuery.value.toLowerCase()))
    })

    const handleBackLibrary = async () => {
       store.setCurrentPageId('YOUR_LIBRARY')
       await router.push({ name: 'YOUR_LIBRARY' })
    }

    const handleAddToLibrary = () => {
       store.playlist_added_to_library = true
    }

    const handleShare = async () => {
       store.setCurrentPageId('PLAYLIST_SHARE')
       await router.push({ name: 'PLAYLIST_SHARE' })
    }

    const handleSearch = () => {
       store.playlist_detail_has_searched = true
    }

    const handleClickTrack = async (track) => {
       store.selected_track_id = track.id
       store.playlist_detail_viewport_anchor_id = null
       store.setCurrentPageId('TRACK_DETAIL')
       await router.push({ name: 'TRACK_DETAIL', params: { id: track.id } })
    }

    const handleClickMatchedTrack = async (track) => {
       store.selected_track_id = track.id
       store.playlist_detail_has_searched = null
       store.setCurrentPageId('TRACK_DETAIL')
       await router.push({ name: 'TRACK_DETAIL', params: { id: track.id } })
    }

    return {
       playlist,
       tracks,
       searchQuery,
       hasSearched,
       searchResults,
       handleBackLibrary,
       handleAddToLibrary,
       handleShare,
       handleSearch,
       handleClickTrack,
       handleClickMatchedTrack
    }
  }
}
</script>