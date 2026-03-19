<template>
  <div class="flex h-screen bg-black text-white font-sans overflow-hidden">
    <aside class="w-64 bg-black flex-shrink-0 p-6 border-r border-[#282828] hidden md:block">
      <div id="back-home" @click="handleBackHome" class="flex items-center space-x-2 text-[#B3B3B3] hover:text-white cursor-pointer font-bold mb-8">
         <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M12.5 3.247a1 1 0 0 0-1 0L4 8.75v9a1 1 0 0 0 1 1h5v-5h4v5h5a1 1 0 0 0 1-1v-9l-7.5-5.503z"/></svg>
         <span>Home</span>
      </div>
      <h1 class="text-3xl font-bold mb-4">Search</h1>
    </aside>

    <main class="flex-1 flex flex-col relative bg-[#121212] overflow-hidden">
       <!-- Header -->
       <header class="h-24 flex items-center px-8 sticky top-0 bg-[#121212] z-30">
          <div class="relative w-full max-w-2xl">
            <input 
              id="global-search-input"
              v-model="searchQuery"
              @keyup.enter="handleSearch"
              class="w-full bg-[#282828] text-white rounded-full pl-12 pr-4 py-3 text-base focus:outline-none focus:ring-2 focus:ring-white transition-all placeholder-[#B3B3B3]"
              placeholder="What do you want to listen to?"
            />
            <svg class="absolute left-4 top-3.5 w-6 h-6 text-[#B3B3B3]" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
          </div>
       </header>

       <!-- Filters Row -->
       <div class="px-8 pb-4 flex items-center space-x-3">
          <div id="search-filter-songs" @click="handleFilterCheckbox" class="bg-[#282828] hover:bg-[#3E3E3E] px-4 py-1.5 rounded-full cursor-pointer text-sm font-bold transition-colors" :class="{'bg-white text-black hover:bg-white': filtersApplied}">
             Songs
          </div>

          <div id="search-sort-dropdown" class="relative group">
             <div class="bg-[#282828] hover:bg-[#3E3E3E] px-4 py-1.5 rounded-full cursor-pointer text-sm font-bold transition-colors flex items-center space-x-1">
                <span>Sort</span>
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
             </div>
             <div class="hidden group-hover:block absolute top-full left-0 mt-2 w-40 bg-[#282828] rounded shadow-xl z-50 border border-[#3E3E3E]">
                <div id="search-sort-option-relevance" class="px-4 py-2 hover:bg-[#3E3E3E] cursor-pointer text-sm" @click="handleSort('relevance')">Relevance</div>
                <div id="search-sort-option-recent" class="px-4 py-2 hover:bg-[#3E3E3E] cursor-pointer text-sm" @click="handleSort('recent')">Recent</div>
             </div>
          </div>
       </div>

       <div 
         id="search-results-tracks"
         class="flex-1 overflow-y-auto px-8 pb-32"
       >
         <div v-if="hasSearched || filtersApplied">
            <h2 class="text-2xl font-bold mb-4">Songs</h2>
            <div class="space-y-1">
               <div 
                 v-for="track in tracksResult" 
                 :key="track.id"
                 class="group flex items-center p-2 rounded-md hover:bg-[#282828] cursor-pointer"
                 :class="[
                    `data-id-${track.id}`,
                    hasSearched ? 'track-row-matched' : (filtersApplied ? 'track-row-filtered' : 'track-row-visible')
                 ]"
                 @click="handleClickTrack(track)"
               >
                 <div class="relative w-10 h-10 mr-4 flex-shrink-0">
                    <img :src="track.image" class="w-full h-full object-cover rounded" />
                    <div class="absolute inset-0 bg-black/50 hidden group-hover:flex items-center justify-center">
                       <svg class="w-4 h-4 text-white fill-current" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
                    </div>
                 </div>
                 <div class="flex-1 min-w-0">
                    <div class="font-medium text-white truncate" :class="{'text-[#1DB954]': false}">{{ track.name }}</div>
                    <div class="text-sm text-[#B3B3B3] truncate">{{ track.artist }}</div>
                 </div>
                 <div class="text-sm text-[#B3B3B3] mx-4 hidden sm:block">{{ track.album }}</div>
                 <div class="text-sm text-[#B3B3B3] w-12 text-right font-variant-numeric tabular-nums">{{ track.duration }}</div>
               </div>
            </div>
         </div>
         
         <div v-else class="flex items-center justify-center h-64 text-[#B3B3B3]">
            Start searching for songs, artists, or podcasts.
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
  name: 'SEARCH_PAGE',
  setup() {
    const store = useSignatureStore()
    const dataStore = useDataStore()
    const router = useRouter()

    const searchQuery = ref('')
    const filtersApplied = ref(false)
    const hasSearched = computed(() => store.search_page_has_searched === true)

    const tracksResult = computed(() => {
      let res = dataStore.tracks
      if (searchQuery.value) {
         res = res.filter(t => t.name.toLowerCase().includes(searchQuery.value.toLowerCase()))
      }
      return res
    })

    const handleBackHome = async () => {
      store.setCurrentPageId('HOME')
      await router.push({ name: 'HOME' })
    }

    const handleFilterCheckbox = () => {
      filtersApplied.value = true
      store.search_page_filters_applied = true
    }

    const handleSort = (val) => {
      filtersApplied.value = true
      store.search_page_filters_applied = true
    }

    const handleSearch = () => {
      store.search_page_has_searched = true
    }

    const handleClickTrack = async (track) => {
      store.selected_track_id = track.id
      store.setCurrentPageId('TRACK_DETAIL')
      
      // Clear effects
      store.search_page_has_searched = null
      store.search_page_filters_applied = null
      store.search_page_viewport_anchor_id = null

      await router.push({ name: 'TRACK_DETAIL', params: { id: track.id } })
    }

    return {
      searchQuery,
      filtersApplied,
      hasSearched,
      tracksResult,
      handleBackHome,
      handleFilterCheckbox,
      handleSort,
      handleSearch,
      handleClickTrack
    }
  }
}
</script>