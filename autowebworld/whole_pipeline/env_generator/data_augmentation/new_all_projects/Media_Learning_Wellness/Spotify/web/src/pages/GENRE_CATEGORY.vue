<template>
  <div class="flex h-screen bg-black text-white font-sans overflow-hidden">
    <aside class="w-64 bg-black flex-shrink-0 p-6 border-r border-[#282828] hidden md:block">
      <div id="back-browse" @click="handleBackBrowse" class="flex items-center space-x-2 text-[#B3B3B3] hover:text-white cursor-pointer font-bold mb-8">
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/></svg>
        <span>Back to Browse</span>
      </div>
      <h1 class="text-3xl font-bold mb-4">Pop</h1>
    </aside>

    <main class="flex-1 flex flex-col relative bg-gradient-to-b from-[#8c1932] to-[#121212] overflow-hidden">
      <!-- Header -->
      <header class="h-24 flex items-center justify-between px-8 bg-black/20">
         <div class="flex items-center space-x-4">
           <!-- Filters -->
           <div id="genre-filter-curated" @click="handleFilterCheckbox" class="bg-black/40 hover:bg-black/60 px-4 py-2 rounded-full cursor-pointer transition-colors border border-transparent hover:border-white">
             <span class="font-bold text-sm">Curated Playlists</span>
           </div>

           <div id="genre-sort-dropdown" class="relative group">
              <div class="flex items-center space-x-2 bg-black/40 hover:bg-black/60 px-4 py-2 rounded-full cursor-pointer">
                 <span class="font-bold text-sm">Sort</span>
                 <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
              </div>
              <div class="hidden group-hover:block absolute top-full left-0 mt-2 w-40 bg-[#282828] rounded shadow-xl z-50">
                <div id="genre-sort-option-popular" class="px-4 py-2 hover:bg-[#3E3E3E] cursor-pointer text-sm" @click="handleSort('popular')">Most Popular</div>
                <div id="genre-sort-option-new" class="px-4 py-2 hover:bg-[#3E3E3E] cursor-pointer text-sm" @click="handleSort('new')">Newest</div>
              </div>
           </div>
         </div>

         <!-- Search -->
         <div class="relative">
            <input 
              id="genre-search-input"
              v-model="searchQuery"
              @keyup.enter="handleSearch"
              class="bg-[#3E3E3E] text-white rounded-full pl-10 pr-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-white w-64"
              placeholder="Search in Pop..."
            />
            <svg class="w-4 h-4 text-gray-400 absolute left-3 top-2.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
         </div>
      </header>

      <div 
        id="genre-playlists"
        class="flex-1 overflow-y-auto px-8 pb-32"
      >
        <section v-if="hasSearched">
           <h2 class="text-2xl font-bold mb-6 mt-4">Results</h2>
           <div id="genre-search-results" class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-6">
              <div 
               v-for="pl in searchResults" 
               :key="pl.id"
               class="playlist-row-matched bg-[#181818]/80 p-4 rounded-md hover:bg-[#282828] transition-colors cursor-pointer"
               :class="`data-id-${pl.id}`"
               @click="handleOpenMatchedPlaylist(pl)"
             >
                <img :src="pl.image" class="w-full aspect-square object-cover rounded-md mb-4 shadow-lg" />
                <h3 class="font-bold truncate">{{ pl.name }}</h3>
             </div>
           </div>
        </section>

        <section v-else>
           <h2 class="text-2xl font-bold mb-6 mt-4">Trending in Pop</h2>
           <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-6">
             <div 
               v-for="pl in filteredPlaylists" 
               :key="pl.id"
               class="playlist-card-visible bg-[#181818]/60 p-4 rounded-md hover:bg-[#282828] transition-colors cursor-pointer group"
               :class="[`data-id-${pl.id}`, {'playlist-card-filtered': filtersApplied}]"
               @click="filtersApplied ? handleOpenFilteredPlaylist(pl) : handleOpenPlaylist(pl)"
             >
                <div class="relative aspect-square mb-4">
                  <img :src="pl.image" class="w-full h-full object-cover rounded-md shadow-lg" />
                  <div class="absolute right-2 bottom-2 bg-[#1DB954] rounded-full p-3 opacity-0 group-hover:opacity-100 transform translate-y-2 group-hover:translate-y-0 transition-all shadow-xl">
                     <svg class="w-6 h-6 text-black fill-current" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
                   </div>
                </div>
                <h3 class="font-bold truncate">{{ pl.name }}</h3>
                <p class="text-sm text-[#B3B3B3] line-clamp-2">{{ pl.description }}</p>
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
  name: 'GENRE_CATEGORY',
  setup() {
    const store = useSignatureStore()
    const dataStore = useDataStore()
    const router = useRouter()

    const searchQuery = ref('')
    const filtersApplied = ref(false)
    const sortOption = ref(null)

    const hasSearched = computed(() => store.genre_category_has_searched === true)

    const filteredPlaylists = computed(() => {
      let res = dataStore.playlists.filter(p => p.category_id === 'cat_pop' || p.category_id === 'cat_party') // Mock logic for "Pop"
      // If filtered
      if (filtersApplied.value) {
        res = res.filter(p => p.featured === true)
      }
      return res
    })

    const searchResults = computed(() => {
      if (!searchQuery.value) return []
      return dataStore.playlists.filter(p => p.name.toLowerCase().includes(searchQuery.value.toLowerCase()))
    })

    const handleBackBrowse = async () => {
      store.setCurrentPageId('BROWSE')
      await router.push({ name: 'BROWSE' })
    }

    const handleFilterCheckbox = () => {
      filtersApplied.value = true
      store.genre_category_filters_applied = true
    }

    const handleSort = (val) => {
      sortOption.value = val
      store.genre_category_filters_applied = true
      filtersApplied.value = true
    }

    const handleSearch = () => {
      store.genre_category_has_searched = true
    }

    const handleOpenFilteredPlaylist = async (pl) => {
      store.selected_playlist_id = pl.id
      store.genre_category_filters_applied = null
      store.setCurrentPageId('PLAYLIST_DETAIL')
      await router.push({ name: 'PLAYLIST_DETAIL', params: { id: pl.id } })
    }

    const handleOpenPlaylist = async (pl) => {
      store.selected_playlist_id = pl.id
      store.genre_category_viewport_anchor_id = null
      store.setCurrentPageId('PLAYLIST_DETAIL')
      await router.push({ name: 'PLAYLIST_DETAIL', params: { id: pl.id } })
    }

    const handleOpenMatchedPlaylist = async (pl) => {
      store.selected_playlist_id = pl.id
      store.genre_category_has_searched = null
      store.setCurrentPageId('PLAYLIST_DETAIL')
      await router.push({ name: 'PLAYLIST_DETAIL', params: { id: pl.id } })
    }

    return {
      searchQuery,
      hasSearched,
      filteredPlaylists,
      searchResults,
      filtersApplied,
      handleBackBrowse,
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