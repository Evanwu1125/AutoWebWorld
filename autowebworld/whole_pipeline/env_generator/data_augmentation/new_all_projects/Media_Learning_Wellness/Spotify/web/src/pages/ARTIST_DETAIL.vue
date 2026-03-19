<template>
  <div class="flex h-screen bg-black text-white font-sans overflow-hidden">
    <aside class="w-64 bg-black flex-shrink-0 p-6 border-r border-[#282828] hidden md:block">
      <div id="back-track" @click="handleBackTrack" class="flex items-center space-x-2 text-[#B3B3B3] hover:text-white cursor-pointer font-bold mb-8">
         <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M12.5 3.247a1 1 0 0 0-1 0L4 8.75v9a1 1 0 0 0 1 1h5v-5h4v5h5a1 1 0 0 0 1-1v-9l-7.5-5.503z"/></svg>
         <span>Back to Song</span>
      </div>
    </aside>

    <main class="flex-1 flex flex-col relative bg-[#121212] overflow-hidden">
      <!-- Artist Hero -->
      <div class="h-80 relative bg-cover bg-center" style="background-image: url('/images/photo1766302474.jpg');">
         <div class="absolute inset-0 bg-gradient-to-t from-[#121212] via-transparent to-transparent"></div>
         <div class="absolute bottom-8 left-8">
            <div class="flex items-center space-x-2 mb-2 text-white">
               <svg class="w-5 h-5 fill-[#3D91F4]" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg>
               <span class="text-sm font-bold">Verified Artist</span>
            </div>
            <h1 class="text-6xl md:text-8xl font-bold mb-4 tracking-tighter">The Midnight</h1>
            <p class="text-lg font-medium">1,245,678 monthly listeners</p>
         </div>
      </div>

      <!-- Action Bar -->
      <div class="px-8 py-6 bg-[#121212] flex items-center space-x-6 sticky top-0 z-20">
         <button class="w-14 h-14 bg-[#1DB954] rounded-full flex items-center justify-center hover:scale-105 hover:bg-[#1ed760] transition-transform shadow-lg">
             <svg class="w-7 h-7 text-black fill-current ml-1" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
         </button>
         <button class="uppercase tracking-widest text-xs font-bold border border-[#727272] hover:border-white px-6 py-2 rounded-sm transition-colors">
            Follow
         </button>
         <button class="text-[#B3B3B3] hover:text-white transition-colors">
            <svg class="w-8 h-8" fill="currentColor" viewBox="0 0 24 24"><path d="M12 8c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm0 2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0 6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z"/></svg>
         </button>
      </div>

      <!-- Main Content -->
      <div 
        id="artist-top-tracks" 
        class="flex-1 overflow-y-auto px-8 pb-32"
      >
        <div class="flex items-center justify-between mb-4">
           <h2 class="text-2xl font-bold">Popular</h2>
           <!-- Search -->
           <div class="relative">
             <input 
               id="artist-search-input"
               v-model="searchQuery"
               @keyup.enter="handleSearch"
               class="bg-[#282828] text-white rounded-full pl-8 pr-4 py-1 text-sm focus:outline-none focus:bg-[#3E3E3E] w-48"
               placeholder="Search tracks"
             />
             <svg class="w-4 h-4 text-gray-400 absolute left-2.5 top-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
           </div>
        </div>

        <!-- Filter -->
        <div class="mb-4">
           <div id="artist-filter-popular" @click="handleFilterCheckbox" class="inline-block px-3 py-1 rounded-full text-sm font-bold cursor-pointer transition-colors border border-transparent" :class="{'bg-white text-black': filtersApplied, 'bg-[#282828] text-white hover:bg-[#3E3E3E]': !filtersApplied}">
              Top Songs Only
           </div>
        </div>

        <div v-if="hasSearched">
           <div id="artist-search-results" class="space-y-1">
             <div 
               v-for="(track, idx) in searchResults" 
               :key="track.id"
               class="track-row-matched group flex items-center p-2 rounded-md hover:bg-[#ffffff1a] cursor-pointer"
               :class="`data-id-${track.id}`"
               @click="handleClickMatchedTrack(track)"
             >
                <div class="w-8 text-center text-[#B3B3B3] group-hover:hidden">{{ idx + 1 }}</div>
                <div class="w-8 text-center hidden group-hover:block"><svg class="w-4 h-4 fill-white" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg></div>
                <img :src="track.image" class="w-10 h-10 mr-4 rounded" />
                <div class="flex-1 font-medium text-white">{{ track.name }}</div>
                <div class="text-[#B3B3B3] text-sm">{{ track.duration }}</div>
             </div>
           </div>
        </div>

        <div v-else class="space-y-1">
           <div 
             v-for="(track, idx) in filteredTracks" 
             :key="track.id"
             class="group flex items-center p-2 rounded-md hover:bg-[#ffffff1a] cursor-pointer"
             :class="[
                `data-id-${track.id}`, 
                filtersApplied ? 'track-row-filtered' : 'track-row-visible'
             ]"
             @click="filtersApplied ? handleClickFilteredTrack(track) : handleClickTrack(track)"
           >
              <div class="w-8 text-center text-[#B3B3B3] group-hover:hidden">{{ idx + 1 }}</div>
              <div class="w-8 text-center hidden group-hover:block"><svg class="w-4 h-4 fill-white" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg></div>
              <img :src="track.image" class="w-10 h-10 mr-4 rounded" />
              <div class="flex-1 font-medium text-white">{{ track.name }}</div>
              <div class="text-[#B3B3B3] text-sm">{{ track.duration }}</div>
           </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'
import { useRouter } from 'vue-router'

export default {
  name: 'ARTIST_DETAIL',
  setup() {
    const store = useSignatureStore()
    const dataStore = useDataStore()
    const router = useRouter()

    const searchQuery = ref('')
    const filtersApplied = ref(false)
    const hasSearched = computed(() => store.artist_top_tracks_has_searched === true)

    // Mock artist ID
    const artistId = 'art_1'
    const tracks = computed(() => dataStore.tracks.filter(t => t.artist.includes('Midnight'))) // Mock

    const filteredTracks = computed(() => {
       if (filtersApplied.value) {
          // Mock filter logic: take first 5
          return tracks.value.slice(0, 5)
       }
       return tracks.value
    })

    const searchResults = computed(() => {
       if (!searchQuery.value) return []
       return tracks.value.filter(t => t.name.toLowerCase().includes(searchQuery.value.toLowerCase()))
    })

    const handleBackTrack = async () => {
       store.setCurrentPageId('TRACK_DETAIL')
       await router.push({ name: 'TRACK_DETAIL' })
    }

    const handleFilterCheckbox = () => {
       filtersApplied.value = true
       store.artist_top_tracks_filters_applied = true
    }

    const handleSearch = () => {
       store.artist_top_tracks_has_searched = true
    }

    const handleClickTrack = async (track) => {
       store.selected_track_id = track.id
       store.artist_top_tracks_viewport_anchor_id = null
       store.setCurrentPageId('TRACK_DETAIL')
       await router.push({ name: 'TRACK_DETAIL', params: { id: track.id } })
    }

    const handleClickFilteredTrack = async (track) => {
       store.selected_track_id = track.id
       store.artist_top_tracks_filters_applied = null
       store.setCurrentPageId('TRACK_DETAIL')
       await router.push({ name: 'TRACK_DETAIL', params: { id: track.id } })
    }

    const handleClickMatchedTrack = async (track) => {
       store.selected_track_id = track.id
       store.artist_top_tracks_has_searched = null
       store.setCurrentPageId('TRACK_DETAIL')
       await router.push({ name: 'TRACK_DETAIL', params: { id: track.id } })
    }

    return {
       searchQuery,
       filtersApplied,
       hasSearched,
       filteredTracks,
       searchResults,
       handleBackTrack,
       handleFilterCheckbox,
       handleSearch,
       handleClickTrack,
       handleClickFilteredTrack,
       handleClickMatchedTrack
    }
  }
}
</script>