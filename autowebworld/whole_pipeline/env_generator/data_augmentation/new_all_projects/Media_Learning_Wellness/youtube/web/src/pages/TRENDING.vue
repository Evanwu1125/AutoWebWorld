<template>
  <div class="min-h-screen bg-[#0F0F0F] text-white flex flex-col">
    <!-- Permissions Modal -->
    <LocationPermissionModal />

    <!-- Navbar (Simplified for sub-pages) -->
    <nav class="sticky top-0 z-50 bg-[#0F0F0F]/95 backdrop-blur border-b border-gray-800 px-4 h-14 flex items-center justify-between">
      <div class="flex items-center gap-4">
        <div id="logo-home" @click="goHome" class="flex items-center gap-1 cursor-pointer">
          <div class="bg-red-600 text-white rounded-lg p-1">
            <svg class="w-6 h-6 fill-current" viewBox="0 0 24 24"><path d="M19.615 3.184c-3.604-.246-11.631-.245-15.23 0-3.897.266-4.356 2.62-4.385 8.816.029 6.185.484 8.549 4.385 8.816 3.6.245 11.626.246 15.23 0 3.897-.266 4.356-2.62 4.385-8.816-.029-6.185-.484-8.549-4.385-8.816zm-10.615 12.816v-8l8 3.993-8 4.007z"/></svg>
          </div>
          <span class="text-xl font-bold tracking-tight">Trending</span>
        </div>
      </div>
      <div class="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center text-sm font-bold">U</div>
    </nav>

    <main class="flex-1 max-w-7xl mx-auto w-full p-4 md:p-6">
      <div class="mb-8">
        <h1 class="text-3xl font-bold mb-2">Trending Videos</h1>
        <p class="text-gray-400">See what's popular in your area right now.</p>
      </div>

      <!-- Filters Toolbar -->
      <div class="bg-[#1F1F1F] rounded-xl p-4 mb-8 border border-gray-800 flex flex-wrap gap-6 items-center">
        <!-- Live Checkbox -->
        <div 
          id="filter-live-checkbox" 
          @click="toggleLiveFilter"
          class="flex items-center gap-2 cursor-pointer select-none"
        >
          <div class="w-5 h-5 rounded border border-gray-500 flex items-center justify-center" :class="{'bg-red-600 border-red-600': isLiveFilter}">
            <svg v-if="isLiveFilter" class="w-3.5 h-3.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7"></path></svg>
          </div>
          <span class="text-sm font-medium">Live Now</span>
        </div>

        <!-- Duration Slider -->
        <div class="flex items-center gap-3 flex-1 min-w-[200px]">
          <span class="text-sm font-medium whitespace-nowrap">Min Duration: {{ durationFilter }}s</span>
          <input 
            id="filter-duration-slider"
            type="range" 
            min="0" 
            max="3600" 
            step="60"
            v-model.number="durationFilter"
            @input="applyFilters"
            class="w-full h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer accent-red-600"
          >
        </div>

        <!-- Sort Dropdown -->
        <div class="relative">
          <div 
            id="sort-dropdown"
            @click="isSortOpen = !isSortOpen"
            class="flex items-center gap-2 bg-[#272727] hover:bg-[#333] px-3 py-1.5 rounded-lg cursor-pointer transition-colors border border-gray-700"
          >
            <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 4h13M3 8h9m-9 4h6m4 0l4-4m0 0l4 4m-4-4v12"></path></svg>
            <span class="text-sm">{{ sortLabel }}</span>
          </div>

          <div 
            v-if="isSortOpen"
            class="absolute right-0 mt-2 w-48 bg-[#272727] rounded-xl shadow-xl border border-gray-700 p-1 z-20"
          >
            <div 
              id="sort-option-view-count-desc" 
              @click="setSort('view_count', 'Most Views')"
              class="px-3 py-2 hover:bg-gray-700 rounded-lg cursor-pointer text-sm"
            >
              Most Views
            </div>
            <div 
              id="sort-option-upload-date-inc" 
              @click="setSort('upload_date', 'Newest')"
              class="px-3 py-2 hover:bg-gray-700 rounded-lg cursor-pointer text-sm"
            >
              Newest
            </div>
            <div 
              id="sort-option-rating" 
              @click="setSort('rating', 'Top Rated')"
              class="px-3 py-2 hover:bg-gray-700 rounded-lg cursor-pointer text-sm"
            >
              Top Rated
            </div>
          </div>
        </div>
      </div>

      <!-- Video List -->
      <div 
        id="trending-list" 
        class="space-y-4 overflow-y-auto max-h-[calc(100vh-250px)] pr-2 custom-scrollbar"
      >
        <div 
          v-for="(video, index) in filteredVideos" 
          :key="video.id"
          class="flex flex-col sm:flex-row gap-4 p-3 hover:bg-[#1F1F1F] rounded-xl transition-colors cursor-pointer group"
          :class="getRowClass(video)"
          :data-id="video.id"
          @click="openVideo(video.id)"
        >
          <!-- Thumbnail -->
          <div class="relative w-full sm:w-64 aspect-video rounded-xl overflow-hidden bg-gray-800 flex-shrink-0">
            <img :src="video.image" :alt="video.title" class="w-full h-full object-cover transform group-hover:scale-105 transition-transform duration-300">
            <div class="absolute bottom-1 right-1 bg-black/80 text-white text-xs px-1.5 py-0.5 rounded">
              {{ formatDuration(video.duration) }}
            </div>
          </div>
          
          <!-- Content -->
          <div class="flex-1 py-1">
            <h3 class="text-lg font-bold line-clamp-2 mb-1 group-hover:text-blue-400 transition-colors">{{ video.title }}</h3>
            <div class="text-sm text-gray-400 mb-2">{{ video.channel }} • {{ video.views }} views • {{ video.date }}</div>
            <p class="text-sm text-gray-500 line-clamp-2 hidden sm:block">
              Experience the trending sensation that everyone is talking about. This video has captured the attention of millions worldwide.
            </p>
          </div>
          
          <!-- Rank Number (Trending Style) -->
          <div class="hidden lg:flex items-center justify-center w-12 text-4xl font-black text-gray-800 italic">
            #{{ index + 1 }}
          </div>
        </div>
        
        <!-- Empty State -->
        <div v-if="filteredVideos.length === 0" class="text-center py-20 text-gray-500">
          <p class="text-xl">No videos found matching your filters.</p>
          <button @click="resetFilters" class="mt-4 text-blue-500 hover:underline">Reset Filters</button>
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
import LocationPermissionModal from '../components/LocationPermissionModal.vue'

export default {
  name: 'TRENDING',
  components: { LocationPermissionModal },
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    // Filter State
    const isLiveFilter = ref(false)
    const durationFilter = ref(0)
    const currentSort = ref(null)
    const sortLabel = ref('Sort By')
    const isSortOpen = ref(false)

    // Filter Logic
    const toggleLiveFilter = () => {
      isLiveFilter.value = !isLiveFilter.value
      store.trending_filters_applied = true
    }

    const applyFilters = () => {
      store.trending_filters_applied = true
    }

    const parseViews = (views) => {
      if (!views) return 0
      const normalized = views.toString().toUpperCase().replace(/\s*VIEWS?/, '')
      if (normalized.endsWith('M')) return parseFloat(normalized) * 1_000_000
      if (normalized.endsWith('K')) return parseFloat(normalized) * 1_000
      return parseFloat(normalized) || 0
    }

    // Parse relative date strings like "2 days ago", "1 week ago", "3 months ago", fallback to 0 (most recent)
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
      store.trending_filters_applied = true
    }

    const resetFilters = () => {
      isLiveFilter.value = false
      durationFilter.value = 0
      currentSort.value = null
      sortLabel.value = 'Sort By'
    }

    const filteredVideos = computed(() => {
      let result = [...dataStore.videos]

      // Filter: Duration (Items with duration > filter)
      if (durationFilter.value > 0) {
        result = result.filter(v => v.duration > durationFilter.value)
      }

      // Filter: Live (Simulated - filter even IDs as "live" for demo)
      if (isLiveFilter.value) {
        result = result.filter((v, i) => i % 2 === 0) 
      }

      // Sort
      if (currentSort.value) {
        if (currentSort.value === 'view_count') {
          result.sort((a, b) => parseViews(b.views) - parseViews(a.views))
        } else if (currentSort.value === 'upload_date') {
          // Newest first (smaller days-ago weight means more recent)
          result.sort((a, b) => parseDateWeight(a.date) - parseDateWeight(b.date))
        } else if (currentSort.value === 'rating') {
          result.sort((a, b) => 0.5 - Math.random()) // Random shuffle for rating
        }
      }

      return result
    })

    const getRowClass = (video) => {
      const classes = [`data-id-${video.id}`]
      
      // If filters are active, use filtered class
      if (store.trending_filters_applied) {
        classes.push('video-row-filtered')
      } else {
        // Otherwise use visible class (for "open any video")
        classes.push('video-row-visible')
      }
      
      return classes.join(' ')
    }

    const formatDuration = (seconds) => {
      const m = Math.floor(seconds / 60)
      const s = seconds % 60
      return `${m}:${s.toString().padStart(2, '0')}`
    }

    // Actions
    const goHome = () => {
      store.currentPageId = 'HOME'
      router.push({ name: 'HOME' })
    }

    const openVideo = (id) => {
      store.selected_video_id = id
      store.trending_viewport_anchor_id = id // Simulate scroll anchor effect
      store.trending_filters_applied = null // Clear filter flag on nav
      store.currentPageId = 'WATCH_VIDEO'
      router.push({ name: 'WATCH_VIDEO', params: { id } })
    }

    return {
      store,
      isLiveFilter,
      durationFilter,
      sortLabel,
      isSortOpen,
      filteredVideos,
      toggleLiveFilter,
      applyFilters,
      setSort,
      resetFilters,
      getRowClass,
      formatDuration,
      goHome,
      openVideo
    }
  }
}
</script>

<style scoped>
.custom-scrollbar::-webkit-scrollbar {
  width: 8px;
}
.custom-scrollbar::-webkit-scrollbar-track {
  background: #1f1f1f;
}
.custom-scrollbar::-webkit-scrollbar-thumb {
  background: #444;
  border-radius: 4px;
}
.custom-scrollbar::-webkit-scrollbar-thumb:hover {
  background: #555;
}
</style>
