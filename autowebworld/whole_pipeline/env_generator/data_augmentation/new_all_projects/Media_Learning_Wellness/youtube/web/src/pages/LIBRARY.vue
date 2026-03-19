<template>
  <div class="min-h-screen bg-[#0F0F0F] text-white flex flex-col">
    <!-- Navbar -->
    <nav class="sticky top-0 z-50 bg-[#0F0F0F]/95 backdrop-blur border-b border-gray-800 px-4 h-14 flex items-center justify-between">
      <div class="flex items-center gap-4">
        <div id="logo-home" @click="goHome" class="flex items-center gap-1 cursor-pointer">
          <div class="bg-red-600 text-white rounded-lg p-1">
            <svg class="w-6 h-6 fill-current" viewBox="0 0 24 24"><path d="M19.615 3.184c-3.604-.246-11.631-.245-15.23 0-3.897.266-4.356 2.62-4.385 8.816.029 6.185.484 8.549 4.385 8.816 3.6.245 11.626.246 15.23 0 3.897-.266 4.356-2.62 4.385-8.816-.029-6.185-.484-8.549-4.385-8.816zm-10.615 12.816v-8l8 3.993-8 4.007z"/></svg>
          </div>
          <span class="text-xl font-bold tracking-tight">Library</span>
        </div>
      </div>
      
      <!-- Search -->
      <div class="flex-1 max-w-xl mx-4">
        <div class="flex w-full group">
           <input 
            id="library-search-input"
            v-model="searchQuery"
            @keyup.enter="performSearch"
            type="text"
            placeholder="Search playlists"
            class="w-full bg-[#121212] border border-gray-700 rounded-full px-4 py-2 text-white focus:border-blue-500 focus:outline-none"
          >
        </div>
      </div>
      
      <div class="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center text-sm font-bold">U</div>
    </nav>

    <main class="flex-1 max-w-7xl mx-auto w-full p-4 md:p-6">
      <div class="flex items-center justify-between mb-8">
        <h1 class="text-2xl font-bold">Your Playlists</h1>
        <button 
          id="create-playlist-button" 
          @click="goCreatePlaylist"
          class="flex items-center gap-2 bg-[#3EA6FF] text-black px-4 py-2 rounded-full font-bold hover:bg-blue-400 transition-colors"
        >
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"></path></svg>
          New Playlist
        </button>
      </div>

      <!-- Filters Toolbar -->
      <div class="flex flex-wrap items-center gap-6 mb-8 bg-[#1F1F1F] p-4 rounded-xl border border-gray-800">
        <h2 class="text-lg font-bold mr-2">Filter</h2>
        
        <!-- Watch Later Checkbox -->
        <div 
          id="filter-watch-later-checkbox" 
          @click="toggleWatchLaterFilter"
          class="flex items-center gap-2 cursor-pointer select-none px-3 py-1.5 rounded-lg hover:bg-[#333] transition-colors"
        >
          <div class="w-5 h-5 rounded border border-gray-500 flex items-center justify-center" :class="{'bg-red-600 border-red-600': isWatchLaterFilter}">
            <svg v-if="isWatchLaterFilter" class="w-3.5 h-3.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7"></path></svg>
          </div>
          <span class="text-sm">Watch Later</span>
        </div>

        <!-- Size Slider -->
        <div class="flex items-center gap-3">
          <span class="text-sm text-gray-400">Min Videos: {{ sizeFilter }}</span>
          <input 
            id="filter-playlist-size-slider"
            type="range" 
            min="0" 
            max="50" 
            step="5"
            v-model.number="sizeFilter"
            @input="applyFilters"
            class="w-32 h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer accent-red-600"
          >
        </div>

        <!-- Sort -->
        <div class="relative ml-auto">
          <div 
            id="library-sort-dropdown"
            @click="isSortOpen = !isSortOpen"
            class="flex items-center gap-2 cursor-pointer hover:text-white text-sm font-medium text-gray-400"
          >
            <span>{{ sortLabel }}</span>
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
          </div>
          
          <div 
            v-if="isSortOpen"
            class="absolute right-0 mt-2 w-40 bg-[#272727] rounded-lg shadow-xl border border-gray-700 py-1 z-20"
          >
            <div id="library-sort-option-recent" @click="setSort('recent', 'Recently Added')" class="px-4 py-2 hover:bg-gray-700 cursor-pointer text-sm">Recently Added</div>
            <div id="library-sort-option-a-z" @click="setSort('a_z', 'A-Z')" class="px-4 py-2 hover:bg-gray-700 cursor-pointer text-sm">A-Z</div>
            <div id="library-sort-option-z-a" @click="setSort('z_a', 'Z-A')" class="px-4 py-2 hover:bg-gray-700 cursor-pointer text-sm">Z-A</div>
          </div>
        </div>
      </div>

      <!-- Grid Layout -->
      <div 
        id="library-playlists" 
        class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6"
      >
        <div 
          v-for="playlist in filteredPlaylists" 
          :key="playlist.id"
          class="flex flex-col cursor-pointer group"
          :class="getRowClass(playlist)"
          :data-id="playlist.id"
          @click="openPlaylist(playlist)"
        >
          <div class="aspect-video bg-gray-800 rounded-xl mb-3 overflow-hidden relative group-hover:shadow-lg transition-all group-hover:-translate-y-1">
             <img :src="playlist.image" :alt="playlist.title" class="w-full h-full object-cover">
             <!-- Playlist Overlay Effect -->
             <div class="absolute right-0 top-0 bottom-0 w-1/3 bg-black/60 flex items-center justify-center flex-col text-white">
                <span class="font-bold text-lg">{{ playlist.count }}</span>
                <svg class="w-5 h-5 mt-1" fill="currentColor" viewBox="0 0 24 24"><path d="M3 13h2v-2H3v2zm0 4h2v-2H3v2zm0-8h2V7H3v2zm4 4h14v-2H7v2zm0 4h14v-2H7v2zM7 7v2h14V7H7z"/></svg>
             </div>
          </div>
          
          <h3 class="font-bold text-sm mb-1 group-hover:text-white text-gray-200 line-clamp-1">{{ playlist.title }}</h3>
          <p class="text-xs text-gray-500 uppercase font-semibold">Playlist</p>
        </div>
      </div>
      
      <!-- Empty State -->
      <div v-if="filteredPlaylists.length === 0" class="text-center py-20 text-gray-500">
        <p class="text-xl">No playlists found.</p>
      </div>
    </main>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'LIBRARY',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    // UI State
    const searchQuery = ref('')
    const isWatchLaterFilter = ref(false)
    const sizeFilter = ref(0)
    const currentSort = ref(null)
    const sortLabel = ref('Default')
    const isSortOpen = ref(false)

    // Actions
    const performSearch = () => {
      if (!searchQuery.value.trim()) return
      store.library_has_searched = true
      const match = dataStore.playlists.find(p => p.title.toLowerCase().includes(searchQuery.value.toLowerCase()))
      store.matched_playlist_id = match ? match.id : null
    }

    const toggleWatchLaterFilter = () => {
      isWatchLaterFilter.value = !isWatchLaterFilter.value
      store.library_filters_applied = true
    }

    const applyFilters = () => {
      store.library_filters_applied = true
    }

    const setSort = (value, label) => {
      currentSort.value = value
      sortLabel.value = label
      isSortOpen.value = false
      store.library_filters_applied = true
    }

    const filteredPlaylists = computed(() => {
      let result = [...dataStore.playlists]

      // Search
      if (searchQuery.value) {
        const q = searchQuery.value.toLowerCase()
        result = result.filter(p => p.title.toLowerCase().includes(q))
      }

      // Size Filter
      if (sizeFilter.value > 0) {
        result = result.filter(p => p.count >= sizeFilter.value)
      }

      // Watch Later Filter (Mock: filter specific id or keyword)
      if (isWatchLaterFilter.value) {
        result = result.filter(p => p.title.toLowerCase().includes('watch later'))
      }

      // Sort
      if (currentSort.value === 'a_z') {
        result.sort((a, b) => a.title.localeCompare(b.title))
      } else if (currentSort.value === 'z_a') {
        result.sort((a, b) => b.title.localeCompare(a.title))
      }

      return result
    })

    const getRowClass = (playlist) => {
      const classes = [`data-id-${playlist.id}`]
      
      if (store.library_filters_applied) {
        classes.push('playlist-row-filtered')
      } else if (store.library_has_searched && store.matched_playlist_id === playlist.id) {
        classes.push('playlist-row-matched')
      } else {
        classes.push('playlist-row-visible')
      }
      
      return classes.join(' ')
    }

    const goHome = () => {
      store.currentPageId = 'HOME'
      router.push({ name: 'HOME' })
    }

    const goCreatePlaylist = () => {
      store.currentPageId = 'PLAYLIST_CREATE_FORM'
      router.push({ name: 'PLAYLIST_CREATE_FORM' })
    }

    const openPlaylist = (playlist) => {
      store.selected_playlist_id = playlist.id
      store.library_viewport_anchor_id = playlist.id
      store.library_filters_applied = null
      store.library_has_searched = null
      store.currentPageId = 'PLAYLIST_DETAIL'
      router.push({ name: 'PLAYLIST_DETAIL', params: { id: playlist.id } })
    }

    return {
      store,
      searchQuery,
      isWatchLaterFilter,
      sizeFilter,
      sortLabel,
      isSortOpen,
      filteredPlaylists,
      performSearch,
      toggleWatchLaterFilter,
      applyFilters,
      setSort,
      getRowClass,
      goHome,
      goCreatePlaylist,
      openPlaylist
    }
  }
}
</script>