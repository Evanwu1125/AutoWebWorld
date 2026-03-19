<template>
  <div class="flex h-screen bg-black text-white font-sans overflow-hidden">
    <aside class="w-64 bg-black flex-shrink-0 p-6 border-r border-[#282828] hidden md:block">
      <div id="back-home" @click="handleBackHome" class="flex items-center space-x-2 text-[#B3B3B3] hover:text-white cursor-pointer font-bold mb-8">
         <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M12.5 3.247a1 1 0 0 0-1 0L4 8.75v9a1 1 0 0 0 1 1h5v-5h4v5h5a1 1 0 0 0 1-1v-9l-7.5-5.503z"/></svg>
         <span>Home</span>
      </div>
      <h1 class="text-3xl font-bold mb-4">Your Library</h1>
      <nav class="space-y-4">
        <div class="text-[#B3B3B3] hover:text-white cursor-pointer">Playlists</div>
        <div class="text-[#B3B3B3] hover:text-white cursor-pointer">Artists</div>
        <div class="text-[#B3B3B3] hover:text-white cursor-pointer">Albums</div>
      </nav>
    </aside>

    <main class="flex-1 flex flex-col relative bg-[#121212] overflow-hidden">
       <!-- Header -->
       <header class="h-20 flex items-center justify-between px-8 bg-[#121212] sticky top-0 z-20 shadow-sm border-b border-[#282828]">
          <div class="flex items-center space-x-4">
             <!-- Filters -->
             <div id="library-filter-downloads" @click="handleFilterCheckbox" class="px-4 py-2 bg-[#282828] hover:bg-[#3E3E3E] rounded-full cursor-pointer transition-colors border border-transparent" :class="{'border-white': filtersApplied}">
               <span class="text-sm font-bold">Downloaded</span>
             </div>
             
             <!-- Sort -->
             <div id="library-sort-dropdown" class="relative group">
                <div class="flex items-center space-x-2 px-4 py-2 bg-[#282828] hover:bg-[#3E3E3E] rounded-full cursor-pointer">
                   <span class="text-sm font-bold">Sort by</span>
                   <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
                </div>
                <div class="hidden group-hover:block absolute top-full left-0 mt-2 w-48 bg-[#282828] rounded shadow-xl z-50 border border-[#3E3E3E]">
                  <div id="library-sort-option-desc" class="px-4 py-3 hover:bg-[#3E3E3E] cursor-pointer text-sm" @click="handleSort('recently_played')">Recently Played</div>
                  <div id="library-sort-option-alpha-inc" class="px-4 py-3 hover:bg-[#3E3E3E] cursor-pointer text-sm" @click="handleSort('alphabetical')">Alphabetical</div>
                </div>
             </div>
          </div>

          <!-- Search -->
          <div class="relative group">
             <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <svg class="h-5 w-5 text-gray-400 group-focus-within:text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
             </div>
             <input 
               id="library-search-input"
               v-model="searchQuery"
               @keyup.enter="handleSearch"
               class="bg-[#282828] text-white rounded-full pl-10 pr-4 py-2 text-sm focus:outline-none focus:bg-[#3E3E3E] transition-all w-64"
               placeholder="Search in Your Library"
             />
          </div>
       </header>

       <div 
         id="library-playlists"
         class="flex-1 overflow-y-auto p-8 pb-32"
       >
         <section v-if="hasSearched">
            <h2 class="text-xl font-bold mb-4">Search Results</h2>
            <div id="library-search-results" class="grid grid-cols-1 gap-2">
               <div 
                 v-for="pl in searchResults" 
                 :key="pl.id"
                 class="playlist-row-matched flex items-center p-2 rounded-md hover:bg-[#282828] cursor-pointer group"
                 :class="`data-id-${pl.id}`"
                 @click="handleOpenMatchedPlaylist(pl)"
               >
                 <img :src="pl.image" class="w-12 h-12 rounded object-cover mr-4" />
                 <div>
                   <h3 class="font-bold text-white group-hover:underline">{{ pl.name }}</h3>
                   <p class="text-sm text-[#B3B3B3]">Playlist • {{ pl.owner }}</p>
                 </div>
               </div>
            </div>
         </section>

         <section v-else>
            <h2 class="text-xl font-bold mb-4">Playlists</h2>
            <!-- List View -->
            <div class="space-y-2">
               <div 
                 v-for="pl in filteredPlaylists" 
                 :key="pl.id"
                 class="flex items-center p-2 rounded-md hover:bg-[#282828] cursor-pointer group"
                 :class="[
                    `data-id-${pl.id}`, 
                    filtersApplied ? 'playlist-row-filtered' : 'playlist-row-visible'
                 ]"
                 @click="filtersApplied ? handleOpenFilteredPlaylist(pl) : handleOpenPlaylist(pl)"
               >
                 <img :src="pl.image" class="w-12 h-12 rounded object-cover mr-4 shadow-sm" />
                 <div class="flex-1">
                   <h3 class="font-bold text-white group-hover:text-[#1DB954] transition-colors">{{ pl.name }}</h3>
                   <p class="text-sm text-[#B3B3B3]">Playlist • {{ pl.owner }}</p>
                 </div>
                 <div class="text-sm text-[#B3B3B3] hidden md:block mr-8">
                    Dec 20, 2025
                 </div>
                 <div class="opacity-0 group-hover:opacity-100 transition-opacity">
                    <svg class="w-8 h-8 text-[#1DB954] fill-current" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
                 </div>
               </div>
            </div>
         </section>
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
  name: 'YOUR_LIBRARY',
  setup() {
    const store = useSignatureStore()
    const dataStore = useDataStore()
    const router = useRouter()

    const searchQuery = ref('')
    const filtersApplied = ref(false)
    const sortOption = ref(null)

    const hasSearched = computed(() => store.your_library_has_searched === true)

    const filteredPlaylists = computed(() => {
      let res = dataStore.playlists

      // 1. Apply filter first
      if (filtersApplied.value) {
         res = res.filter(p => p.downloaded === true)
      }

      // 2. Apply sort
      if (sortOption.value === 'alphabetical') {
         res = [...res].sort((a, b) => a.name.localeCompare(b.name))
      } else if (sortOption.value === 'recently_played') {
         res = [...res]  // Keep original order or add custom sort logic
      }

      return res
    })

    const searchResults = computed(() => {
      if (!searchQuery.value) return []
      return dataStore.playlists.filter(p => p.name.toLowerCase().includes(searchQuery.value.toLowerCase()))
    })

    const handleBackHome = async () => {
      store.setCurrentPageId('HOME')
      await router.push({ name: 'HOME' })
    }

    const handleFilterCheckbox = () => {
      filtersApplied.value = true
      store.your_library_filters_applied = true
    }

    const handleSort = (val) => {
      sortOption.value = val
      store.your_library_filters_applied = true
    }

    const handleSearch = () => {
      store.your_library_has_searched = true
    }

    const handleOpenFilteredPlaylist = async (pl) => {
      store.selected_playlist_id = pl.id
      store.your_library_filters_applied = null
      store.setCurrentPageId('PLAYLIST_DETAIL')
      await router.push({ name: 'PLAYLIST_DETAIL', params: { id: pl.id } })
    }

    const handleOpenPlaylist = async (pl) => {
      store.selected_playlist_id = pl.id
      store.your_library_viewport_anchor_id = null
      store.setCurrentPageId('PLAYLIST_DETAIL')
      await router.push({ name: 'PLAYLIST_DETAIL', params: { id: pl.id } })
    }

    const handleOpenMatchedPlaylist = async (pl) => {
      store.selected_playlist_id = pl.id
      store.your_library_has_searched = null
      store.setCurrentPageId('PLAYLIST_DETAIL')
      await router.push({ name: 'PLAYLIST_DETAIL', params: { id: pl.id } })
    }

    return {
      searchQuery,
      filtersApplied,
      hasSearched,
      filteredPlaylists,
      searchResults,
      handleBackHome,
      handleFilterCheckbox,
      handleSort,
      handleSearch,
      handleOpenFilteredPlaylist,
      handleOpenPlaylist,
      handleOpenMatchedPlaylist
    }
  }
}
</script>