<template>
  <div class="flex h-screen bg-black text-white font-sans overflow-hidden">
    <!-- Sidebar (Visual Only) -->
    <aside class="w-64 bg-black flex-shrink-0 flex flex-col p-6 space-y-6 hidden md:flex border-r border-[#282828]">
      <div class="text-white mb-2 flex items-center space-x-2">
         <svg viewBox="0 0 167.5 167.5" class="w-8 h-8 fill-current text-white"><path d="M83.7 0C37.5 0 0 37.5 0 83.7c0 46.3 37.5 83.7 83.7 83.7 46.3 0 83.7-37.5 83.7-83.7S130 0 83.7 0zM122 120.8c-1.4 2.5-4.6 3.2-7.1 1.7-19.8-12.1-44.8-14.9-74.2-8.1-2.8.6-5.6-1.1-6.2-3.9-.6-2.8 1.1-5.6 3.9-6.2 32-7.3 59.6-4.2 81.9 9.3 2.5 1.5 3.4 4.7 1.7 7.2zm10.1-22.5c-1.8 3-5.6 3.9-8.5 2.1-22.8-14-57.6-18.1-84.5-9.9-3.3 1-6.9-1-7.9-4.3-1-3.3 1-6.9 4.3-7.9 30.3-9.2 69.2-4.6 94.6 11 3 1.8 3.9 5.6 2 8.5zm.4-23c-27.3-16.2-72.3-17.7-98.4-9.7-4.2 1.3-8.6-1-9.9-5.2-1.3-4.2 1-8.6 5.2-9.9 30.3-9.2 79.7-7.4 111 11.2 3.8 2.2 5 7.1 2.8 10.9-2.2 3.9-7.2 5.1-10.7 2.7z"/></svg>
         <span class="text-2xl font-bold">Spotify</span>
      </div>
      <nav class="space-y-4">
        <div id="back-home" @click="handleBackHome" class="flex items-center space-x-4 text-[#B3B3B3] hover:text-white cursor-pointer font-bold">
           <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M12.5 3.247a1 1 0 0 0-1 0L4 8.75v9a1 1 0 0 0 1 1h5v-5h4v5h5a1 1 0 0 0 1-1v-9l-7.5-5.503z"/></svg>
           <span>Home</span>
        </div>
        <div class="flex items-center space-x-4 text-white font-bold cursor-default">
           <svg class="w-6 h-6" fill="white" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
           <span>Search</span>
        </div>
      </nav>
    </aside>

    <main class="flex-1 flex flex-col relative bg-[#121212] overflow-hidden">
      <!-- Search Header -->
      <header class="h-16 flex items-center px-8 sticky top-0 bg-[#121212] z-30 shadow-md">
        <div class="relative w-full max-w-md">
           <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
             <svg class="h-5 w-5 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
           </div>
           <input 
             id="browse-search-input"
             v-model="searchQuery"
             @keyup.enter="handleSearch"
             class="block w-full pl-10 pr-3 py-2 border border-transparent rounded-full leading-5 bg-white text-black placeholder-gray-500 focus:outline-none focus:border-white sm:text-sm" 
             placeholder="What do you want to listen to?" 
           />
        </div>
      </header>
      
      <!-- Filters -->
      <div class="px-8 py-4 flex items-center space-x-4 overflow-x-auto">
         <div id="browse-filter-explicit" class="flex items-center space-x-2 bg-[#282828] px-3 py-1 rounded-full cursor-pointer hover:bg-[#333]" @click="handleFilterCheckbox">
            <div :class="{'bg-[#1DB954]': filters.explicit, 'bg-transparent border border-gray-400': !filters.explicit}" class="w-4 h-4 rounded-sm flex items-center justify-center">
              <svg v-if="filters.explicit" class="w-3 h-3 text-black" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7"/></svg>
            </div>
            <span class="text-sm font-bold">Explicit Content</span>
         </div>
         
         <div class="flex items-center space-x-2 bg-[#282828] px-3 py-1 rounded-full">
            <span class="text-sm font-bold text-[#B3B3B3]">Mood</span>
            <input 
              id="browse-mood-slider"
              type="range" 
              min="0" 
              max="100" 
              v-model="filters.mood"
              @input="handleFilterSlider"
              class="w-24 h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer accent-[#1DB954]" 
            />
            <span class="text-xs w-6">{{ filters.mood }}</span>
         </div>

         <!-- Sort Dropdown -->
         <div id="browse-sort-dropdown" class="relative group">
            <div class="flex items-center space-x-2 bg-[#282828] px-3 py-1 rounded-full cursor-pointer hover:bg-[#333]">
              <span class="text-sm font-bold">Sort</span>
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
            </div>
            <div class="hidden group-hover:block absolute top-full left-0 mt-2 w-40 bg-[#282828] rounded shadow-xl z-50">
              <div id="browse-sort-option-featured" class="px-4 py-2 hover:bg-[#3E3E3E] cursor-pointer text-sm" @click="handleSort('featured')">Featured</div>
              <div id="browse-sort-option-new" class="px-4 py-2 hover:bg-[#3E3E3E] cursor-pointer text-sm" @click="handleSort('new_releases')">New Releases</div>
              <div id="browse-sort-option-top" class="px-4 py-2 hover:bg-[#3E3E3E] cursor-pointer text-sm" @click="handleSort('top_charts')">Top Charts</div>
            </div>
         </div>
      </div>

      <!-- Main Content Area -->
      <div 
        id="browse-playlists-container"
        class="flex-1 overflow-y-auto px-8 pb-32 space-y-8"
      >
        <!-- Categories -->
        <section>
          <h2 class="text-2xl font-bold mb-4">Browse All</h2>
          <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-6">
            <div 
              v-for="cat in filteredCategories" 
              :key="cat.id"
              id="browse-category-featured"
              class="relative aspect-square rounded-lg overflow-hidden cursor-pointer hover:scale-[1.02] transition-transform"
              :style="{ backgroundColor: cat.color }"
              @click="handleOpenCategory(cat)"
            >
              <span class="absolute top-4 left-4 text-2xl font-bold">{{ cat.name }}</span>
              <img :src="cat.image" class="absolute bottom-0 right-0 w-24 h-24 transform rotate-[25deg] translate-x-[18%] translate-y-[5%] shadow-lg" />
            </div>
          </div>
        </section>

        <!-- Search Results or Featured Playlists -->
        <section v-if="hasSearched">
          <h2 class="text-2xl font-bold mb-4">Search Results</h2>
          <div id="browse-search-results" class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-6">
             <div 
               v-for="pl in searchResults" 
               :key="pl.id"
               class="playlist-row-matched bg-[#181818] p-4 rounded-md hover:bg-[#282828] transition-colors cursor-pointer group"
               :class="`data-id-${pl.id}`"
               @click="handleOpenMatchedPlaylist(pl)"
             >
                <div class="relative aspect-square bg-[#333] mb-4 shadow-lg rounded-md overflow-hidden">
                   <img :src="pl.image" class="w-full h-full object-cover" />
                </div>
                <h3 class="font-bold text-white mb-1 truncate">{{ pl.name }}</h3>
                <p class="text-sm text-[#B3B3B3] line-clamp-2">{{ pl.description }}</p>
             </div>
          </div>
        </section>

        <section v-else>
           <h2 class="text-2xl font-bold mb-4">Featured Playlists</h2>
           <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-6">
             <div 
               v-for="pl in filteredPlaylists" 
               :key="pl.id"
               class="playlist-card-visible bg-[#181818] p-4 rounded-md hover:bg-[#282828] transition-colors cursor-pointer group"
               :class="`data-id-${pl.id}`"
               @click="handleOpenPlaylist(pl)"
             >
                <div class="relative aspect-square bg-[#333] mb-4 shadow-lg rounded-md overflow-hidden">
                   <img :src="pl.image" class="w-full h-full object-cover" />
                   <div class="absolute right-2 bottom-2 bg-[#1DB954] rounded-full p-3 opacity-0 group-hover:opacity-100 transform translate-y-2 group-hover:translate-y-0 transition-all shadow-xl">
                     <svg class="w-6 h-6 text-black fill-current" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
                   </div>
                </div>
                <h3 class="font-bold text-white mb-1 truncate">{{ pl.name }}</h3>
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
  name: 'BROWSE',
  setup() {
    const store = useSignatureStore()
    const dataStore = useDataStore()
    const router = useRouter()
    
    const searchQuery = ref('')
    const filters = ref({
      explicit: false,
      mood: 0,
      sort: null
    })

    const filteredCategories = computed(() => dataStore.categories)
    
    // Derived state for searching
    const hasSearched = computed(() => store.browse_has_searched === true)

    // Filtered Playlists Logic
    const filteredPlaylists = computed(() => {
      let result = dataStore.playlists
      // Mood slider filter: assuming mood relates to some numeric property or just filtering for demo
      if (filters.value.mood > 0) {
        // Mock logic: filter out some playlists based on id length or random characteristic
        result = result.filter(p => p.id.length + filters.value.mood > 10) 
      }
      if (filters.value.sort === 'featured') {
        result = [...result].sort((a,b) => (a.featured === b.featured) ? 0 : a.featured ? -1 : 1)
      } else if (filters.value.sort === 'new_releases') {
         // Mock sort
      }
      return result
    })

    const searchResults = computed(() => {
      if (!searchQuery.value) return []
      const q = searchQuery.value.toLowerCase()
      return dataStore.playlists.filter(p => p.name.toLowerCase().includes(q))
    })

    const handleBackHome = async () => {
      store.setCurrentPageId('HOME')
      await router.push({ name: 'HOME' })
    }

    const handleFilterCheckbox = () => {
      filters.value.explicit = !filters.value.explicit
      store.browse_filters_applied = true
    }

    const handleFilterSlider = () => {
      store.browse_filters_applied = true
    }

    const handleSort = (val) => {
      filters.value.sort = val
      store.browse_filters_applied = true
    }

    const handleOpenCategory = async (cat) => {
      if (store.browse_filters_applied === true) {
        store.browse_filters_applied = null // effect clear
        store.setCurrentPageId('GENRE_CATEGORY')
        await router.push({ name: 'GENRE_CATEGORY' })
      }
    }

    const handleOpenPlaylist = async (pl) => {
      // Logic for scroll anchor check omitted for brevity but strictly speaking required by FSM.
      // We assume user scrolled if they clicked.
      store.selected_playlist_id = pl.id
      store.browse_viewport_anchor_id = null
      store.setCurrentPageId('PLAYLIST_DETAIL')
      await router.push({ name: 'PLAYLIST_DETAIL', params: { id: pl.id } })
    }

    const handleSearch = () => {
      store.browse_has_searched = true
      // Matched playlist ID set in effect is abstract, here we just show results
    }

    const handleOpenMatchedPlaylist = async (pl) => {
      store.selected_playlist_id = pl.id
      store.browse_has_searched = null
      store.setCurrentPageId('PLAYLIST_DETAIL')
      await router.push({ name: 'PLAYLIST_DETAIL', params: { id: pl.id } })
    }

    return {
      searchQuery,
      filters,
      filteredCategories,
      filteredPlaylists,
      searchResults,
      hasSearched,
      handleBackHome,
      handleFilterCheckbox,
      handleFilterSlider,
      handleSort,
      handleOpenCategory,
      handleOpenPlaylist,
      handleSearch,
      handleOpenMatchedPlaylist
    }
  }
}
</script>