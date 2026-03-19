<template>
  <div class="min-h-screen bg-[#0F0F0F] text-white flex flex-col">
    <!-- Navbar -->
    <nav class="sticky top-0 z-50 bg-[#0F0F0F]/95 backdrop-blur border-b border-gray-800 px-4 h-14 flex items-center justify-between">
      <div class="flex items-center gap-4">
        <div id="logo-home" @click="goHome" class="flex items-center gap-1 cursor-pointer">
          <div class="bg-red-600 text-white rounded-lg p-1">
            <svg class="w-6 h-6 fill-current" viewBox="0 0 24 24"><path d="M19.615 3.184c-3.604-.246-11.631-.245-15.23 0-3.897.266-4.356 2.62-4.385 8.816.029 6.185.484 8.549 4.385 8.816 3.6.245 11.626.246 15.23 0 3.897-.266 4.356-2.62 4.385-8.816-.029-6.185-.484-8.549-4.385-8.816zm-10.615 12.816v-8l8 3.993-8 4.007z"/></svg>
          </div>
        </div>
      </div>

      <!-- Search Input (Center) -->
      <div class="flex-1 max-w-2xl mx-4">
        <div class="flex w-full group">
          <div class="relative flex-1">
             <input 
              id="results-search-input"
              type="text"
              v-model="searchQuery"
              @keyup.enter="performSearch"
              placeholder="Search"
              class="w-full bg-[#121212] border border-gray-700 rounded-l-full px-4 py-2 text-white focus:border-blue-500 focus:outline-none placeholder-gray-500"
            >
          </div>
          <button 
            @click="performSearch"
            class="bg-[#222] border border-l-0 border-gray-700 rounded-r-full px-6 hover:bg-[#333] transition-colors"
          >
            <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
          </button>
        </div>
      </div>

      <div class="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center text-sm font-bold">U</div>
    </nav>

    <main class="flex-1 max-w-7xl mx-auto w-full p-4 md:p-6 flex flex-col">
      <!-- Filters Toolbar -->
      <div class="flex flex-wrap items-center gap-4 py-3 border-b border-gray-800 mb-4 sticky top-14 bg-[#0F0F0F] z-10">
        <button 
          id="filter-toggle"
          class="flex items-center gap-2 px-3 py-1.5 rounded-full hover:bg-[#272727] transition-colors text-sm font-medium border border-gray-700"
        >
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"></path></svg>
          Filters
        </button>

        <!-- HD Filter -->
        <div 
          id="filter-hd-checkbox" 
          @click="toggleHDFilter"
          class="px-3 py-1.5 rounded-full cursor-pointer text-sm border transition-colors select-none"
          :class="isHDFilter ? 'bg-white text-black border-white' : 'bg-[#272727] border-gray-700 text-gray-300 hover:bg-[#3f3f3f]'"
        >
          HD Only
        </div>

        <!-- Length Slider -->
        <div class="flex items-center gap-2 bg-[#1F1F1F] px-3 py-1.5 rounded-full border border-gray-700">
          <span class="text-xs text-gray-400">Min Length: {{ lengthFilter }}s</span>
          <input 
            id="filter-length-slider"
            type="range" 
            min="0" 
            max="1800" 
            step="60"
            v-model.number="lengthFilter"
            @input="applyFilters"
            class="w-24 h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer accent-white"
          >
        </div>

        <!-- Sort -->
        <div class="relative ml-auto">
          <div 
            id="search-sort-dropdown"
            @click="isSortOpen = !isSortOpen"
            class="text-sm font-medium text-gray-400 cursor-pointer hover:text-white flex items-center gap-1"
          >
            {{ sortLabel }}
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
          </div>
          <div 
            v-if="isSortOpen"
            class="absolute right-0 mt-2 w-40 bg-[#272727] rounded-lg shadow-xl border border-gray-700 py-1 z-20"
          >
            <div id="search-sort-option-relevance" @click="setSort('relevance', 'Relevance')" class="px-4 py-2 hover:bg-gray-700 cursor-pointer text-sm">Relevance</div>
            <div id="search-sort-option-upload-date" @click="setSort('upload_date', 'Upload Date')" class="px-4 py-2 hover:bg-gray-700 cursor-pointer text-sm">Upload Date</div>
            <div id="search-sort-option-view-count" @click="setSort('view_count', 'View Count')" class="px-4 py-2 hover:bg-gray-700 cursor-pointer text-sm">View Count</div>
          </div>
        </div>
      </div>

      <!-- Results List -->
      <div 
        id="results-list" 
        class="space-y-4 flex-1 overflow-y-auto"
      >
        <div 
          v-for="video in filteredVideos" 
          :key="video.id"
          class="flex flex-col sm:flex-row gap-4 group cursor-pointer"
          :class="getRowClass(video)"
          :data-id="video.id"
          @click="openVideo(video)"
        >
          <!-- Thumbnail -->
          <div class="relative w-full sm:w-[360px] aspect-video rounded-xl overflow-hidden bg-gray-800 flex-shrink-0">
             <img :src="video.image" :alt="video.title" class="w-full h-full object-cover">
             <div class="absolute bottom-1 right-1 bg-black/80 text-white text-xs px-1.5 py-0.5 rounded">
              {{ formatDuration(video.duration) }}
            </div>
          </div>

          <!-- Info -->
          <div class="flex-1 py-1 min-w-0">
            <h3 class="text-lg font-medium mb-1 line-clamp-2 group-hover:text-blue-400 transition-colors">
              {{ video.title }}
            </h3>
            <div class="text-xs text-gray-400 mb-3 flex items-center flex-wrap gap-1">
              <span>{{ video.views }} views</span>
              <span>•</span>
              <span>{{ video.date }}</span>
            </div>
            
            <!-- Channel Info -->
            <div class="flex items-center gap-2 mb-3">
              <div class="w-6 h-6 rounded-full bg-gray-600 overflow-hidden">
                <!-- Using generic avatar image based on name hash logic if available, or just gray -->
              </div>
              <span class="text-xs text-gray-400 hover:text-white transition-colors">{{ video.channel }}</span>
            </div>

            <p class="text-xs text-gray-500 line-clamp-2">
              Description text for the video goes here. This gives a brief overview of the content found in the video.
            </p>

            <div v-if="isHDFilter && video.id.length % 2 === 0" class="mt-2 inline-block bg-gray-800 text-gray-400 text-[10px] px-1 rounded uppercase font-bold tracking-wider">
              HD
            </div>
          </div>
        </div>
        
        <!-- Empty State -->
        <div v-if="filteredVideos.length === 0" class="text-center py-20">
          <div class="w-24 h-24 bg-gray-800 rounded-full flex items-center justify-center mx-auto mb-6 text-gray-600">
            <svg class="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
          </div>
          <h2 class="text-xl font-bold mb-2">No results found</h2>
          <p class="text-gray-400">Try different keywords or remove search filters</p>
        </div>
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
  name: 'SEARCH_RESULTS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    // UI State
    const searchQuery = ref('')
    const isHDFilter = ref(false)
    const lengthFilter = ref(0)
    const currentSort = ref(null)
    const sortLabel = ref('Default')
    const isSortOpen = ref(false)

    // Actions
    const performSearch = () => {
      if (!searchQuery.value.trim()) return
      store.search_results_has_searched = true
      // Matched item ID logic (taking first match for demo)
      const match = dataStore.videos.find(v => v.title.toLowerCase().includes(searchQuery.value.toLowerCase()))
      store.matched_video_id = match ? match.id : null
    }

    const toggleHDFilter = () => {
      isHDFilter.value = !isHDFilter.value
      store.search_results_filters_applied = true
    }

    const applyFilters = () => {
      store.search_results_filters_applied = true
    }

    const parseViews = (views) => {
      if (!views) return 0
      const normalized = views.toString().toUpperCase().replace(/\s*VIEWS?/, '')
      if (normalized.endsWith('M')) return parseFloat(normalized) * 1_000_000
      if (normalized.endsWith('K')) return parseFloat(normalized) * 1_000
      return parseFloat(normalized) || 0
    }

    const parseDateWeight = (dateStr) => {
      if (!dateStr) return Number.POSITIVE_INFINITY
      const normalized = dateStr.toLowerCase().trim()
      const match = normalized.match(/^(\d+)\s*(day|week|month|year)s?\s+ago$/)
      if (!match) return Number.POSITIVE_INFINITY
      const count = parseInt(match[1], 10)
      const unit = match[2]
      const daysPerUnit = { day: 1, week: 7, month: 30, year: 365 }
      return count * (daysPerUnit[unit] || Number.POSITIVE_INFINITY)
    }

    const setSort = (value, label) => {
      currentSort.value = value
      sortLabel.value = label
      isSortOpen.value = false
      store.search_results_filters_applied = true
    }

    const filteredVideos = computed(() => {
      let result = [...dataStore.videos]

      // Search Filter
      if (searchQuery.value) {
        const q = searchQuery.value.toLowerCase()
        result = result.filter(v => 
          v.title.toLowerCase().includes(q) || 
          v.channel.toLowerCase().includes(q)
        )
      }

      // Length Filter (> min)
      if (lengthFilter.value > 0) {
        result = result.filter(v => v.duration > lengthFilter.value)
      }

      // HD Filter (Mock: even IDs are HD)
      if (isHDFilter.value) {
        result = result.filter((v, i) => i % 2 === 0)
      }

      // Sort
      if (currentSort.value === 'view_count') {
         result.sort((a, b) => parseViews(b.views) - parseViews(a.views))
      } else if (currentSort.value === 'upload_date') {
         result.sort((a, b) => parseDateWeight(a.date) - parseDateWeight(b.date)) // Newest first
      }
      
      return result
    })

    const getRowClass = (video) => {
      const classes = [`data-id-${video.id}`]
      
      if (store.search_results_filters_applied) {
        classes.push('video-row-filtered')
      } else if (store.search_results_has_searched && store.matched_video_id === video.id) {
        classes.push('video-row-matched')
      } else {
        classes.push('video-row-visible')
      }
      
      return classes.join(' ')
    }

    const formatDuration = (seconds) => {
      const m = Math.floor(seconds / 60)
      const s = seconds % 60
      return `${m}:${s.toString().padStart(2, '0')}`
    }

    const goHome = () => {
      store.currentPageId = 'HOME'
      router.push({ name: 'HOME' })
    }

    const openVideo = (video) => {
      store.selected_video_id = video.id
      store.search_results_viewport_anchor_id = video.id
      store.search_results_filters_applied = null
      store.search_results_has_searched = null
      store.currentPageId = 'WATCH_VIDEO'
      router.push({ name: 'WATCH_VIDEO', params: { id: video.id } })
    }

    return {
      store,
      searchQuery,
      isHDFilter,
      lengthFilter,
      sortLabel,
      isSortOpen,
      filteredVideos,
      performSearch,
      toggleHDFilter,
      applyFilters,
      setSort,
      getRowClass,
      formatDuration,
      goHome,
      openVideo
    }
  }
}
</script>
