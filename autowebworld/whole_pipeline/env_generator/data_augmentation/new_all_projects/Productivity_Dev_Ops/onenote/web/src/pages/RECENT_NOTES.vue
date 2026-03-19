<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <!-- Navbar -->
    <header class="bg-purple-700 text-white shadow-md z-20 sticky top-0">
      <div class="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
        <div class="flex items-center gap-4">
          <button 
            id="back-home-from-recents" 
            @click="goHome" 
            class="hover:bg-purple-600 p-2 rounded-full transition"
          >
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
          </button>
          <h1 class="text-2xl font-bold flex items-center gap-2">
            <span>🕒</span> Recent Notes
          </h1>
        </div>
      </div>
    </header>

    <!-- Main Content -->
    <main class="flex-1 max-w-7xl mx-auto w-full px-4 py-8">
      
      <!-- Toolbar -->
      <div class="bg-white rounded-xl shadow-sm p-4 mb-8 flex flex-col md:flex-row gap-6 items-center justify-between sticky top-20 z-10">
        <!-- Search -->
        <div class="relative w-full md:w-80">
          <input 
            id="recent-search-input"
            type="text"
            v-model="searchQuery"
            @keyup.enter="handleSearch"
            placeholder="Search recent notes..."
            class="w-full pl-10 pr-4 py-2 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
          />
          <svg class="w-5 h-5 text-gray-400 absolute left-3 top-2.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
        </div>

        <!-- Filters -->
        <div class="flex flex-wrap items-center gap-6">
          <!-- Pinned Checkbox -->
          <label class="flex items-center gap-2 cursor-pointer select-none">
            <div 
              id="recent-filter-pinned-checkbox"
              @click="togglePinnedFilter"
              class="w-5 h-5 border-2 rounded flex items-center justify-center transition-colors"
              :class="pinnedFilter ? 'bg-yellow-500 border-yellow-500' : 'border-gray-300'"
            >
              <svg v-if="pinnedFilter" class="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
            </div>
            <span class="text-sm font-medium text-gray-700">Pinned Only</span>
          </label>

          <!-- Recency Slider (Days Ago) -->
          <div class="flex items-center gap-3">
            <span class="text-sm font-medium text-gray-700">Within {{ recencyFilter }} days</span>
            <input 
              id="recent-recency-slider"
              type="range"
              min="0"
              max="30"
              step="1"
              v-model.number="recencyFilter"
              @input="handleRecencyFilter"
              class="w-32 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-purple-600"
            />
          </div>

          <!-- Sort Dropdown -->
          <div class="relative z-30">
            <button 
              id="recent-sort-dropdown"
              @click="showSortMenu = !showSortMenu"
              class="flex items-center gap-2 bg-gray-100 hover:bg-gray-200 px-4 py-2 rounded-lg text-sm font-medium transition"
            >
              <span>Sort: {{ currentSort }}</span>
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
            </button>
            <div v-if="showSortMenu" class="absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-xl border border-gray-100 py-1 overflow-hidden">
              <div id="recent-sort-option-recent-desc" @click="handleSort('recent')" class="px-4 py-2 hover:bg-purple-50 cursor-pointer text-sm">Most Recent</div>
              <div id="recent-sort-option-title" @click="handleSort('title')" class="px-4 py-2 hover:bg-purple-50 cursor-pointer text-sm">Title</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Recent List -->
      <div id="recent-list-container">
        <div id="recent-list" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div
            v-for="item in filteredData"
            :key="item.id"
            :class="`data-id-${item.id}`"
            class="group rounded-xl shadow-sm hover:shadow-lg transition-all duration-300 border overflow-hidden cursor-pointer"
            :style="item.type === 'page' && item.section ? 'border: 2px solid #93c5fd;' : ''"
            @click="openItem(item)"
          >
            <!-- Section Header (only for pages with section) -->
            <div v-if="item.type === 'page' && item.section"
                 :class="['bg-blue-100 px-3 py-2 border-b border-blue-200', `data-id-${item.section.id}`]">
              <div class="flex items-center gap-2">
                <div class="bg-blue-500 text-white text-xs px-2 py-0.5 rounded font-bold flex items-center gap-1">
                  <span>📂</span> {{ item.section.name }}
                </div>
                <svg v-if="item.section.pinned" class="w-3 h-3 text-blue-600 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20"><path d="M10 2a1 1 0 011 1v1.323l3.954 1.582 1.599-.8a1 1 0 01.894 1.79l-1.233.616 1.738 5.42a1 1 0 01-.285 1.05A3.989 3.989 0 0115 15a3.989 3.989 0 01-2.667-1.019 1 1 0 01-.285-1.05l1.738-5.42-1.233-.617a1 1 0 01.894-1.788l1.599.799L11 4.323V3a1 1 0 011-1zm-5 8.274l-.818 2.552c-.25.78-.03 1.632.57 2.212.617.602 1.504.86 2.342.725L6 18.75a.75.75 0 01-1.5 0V14.5a1 1 0 01-.28-.14l-2.5-2.5a1 1 0 011.06-1.586z"></path></svg>
                <span class="text-xs text-blue-700 ml-auto">Activity: {{ item.section.activity }}%</span>
              </div>
            </div>

            <!-- Image -->
            <div class="h-32 relative overflow-hidden" :class="{
              'bg-gray-200': item.type === 'page',
              'bg-yellow-100': item.type === 'quick_note'
            }">
              <img :src="item.image" :alt="item.title" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" />
              <div class="absolute top-2 right-2 bg-black/50 text-white text-xs px-2 py-1 rounded backdrop-blur-sm">
                {{ item.created_at }}
              </div>
              <!-- Quick Note Badge -->
              <div v-if="item.type === 'quick_note'" class="absolute top-2 left-2 bg-yellow-400 text-yellow-900 text-xs px-2 py-1 rounded font-bold flex items-center gap-1">
                <span>📝</span> Quick
              </div>
            </div>

            <!-- Content -->
            <div class="p-4" :class="{
              'bg-white': item.type === 'page',
              'bg-yellow-50': item.type === 'quick_note'
            }">
              <div class="flex items-center gap-2 mb-2">
                <h3 class="font-bold text-base truncate transition-colors" :class="{
                  'text-gray-900 group-hover:text-purple-700': item.type === 'page',
                  'text-yellow-900 group-hover:text-yellow-700': item.type === 'quick_note'
                }">{{ item.title }}</h3>
                <svg v-if="item.favorite" class="w-4 h-4 text-yellow-500 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path></svg>
              </div>
              <p class="text-sm line-clamp-3 leading-relaxed" :class="{
                'text-gray-500': item.type === 'page',
                'text-yellow-800': item.type === 'quick_note'
              }">{{ item.body }}</p>
            </div>
          </div>
        </div>
      </div>

      <!-- Empty State -->
      <div v-if="filteredData.length === 0" class="text-center py-20 text-gray-500">
        <p class="text-xl">No recent notes found.</p>
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
  name: 'RECENT_NOTES',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    // UI state
    const searchQuery = ref('')
    const pinnedFilter = ref(false)
    const recencyFilter = ref(30) // Default 30 days
    const showSortMenu = ref(false)
    const currentSort = ref(null)

    const hasFilters = computed(() => pinnedFilter.value || recencyFilter.value < 30 || currentSort.value !== 'recent')
    const hasSearched = computed(() => !!store.RECENT_NOTES_matched_page_id)

    const filteredData = computed(() => {
      // Create items array: each page with its section info, plus quick notes
      let items = []

      // Add pages with section wrapper info
      dataStore.pages.forEach(page => {
        const section = dataStore.sections.find(sec => sec.id === page.section_id)
        if (section) {
          items.push({
            ...page,
            type: 'page',
            section: {
              id: section.id,
              name: section.name,
              activity: section.activity,
              pinned: section.pinned
            }
          })
        }
      })

      // Add quick notes (no section wrapper)
      dataStore.quick_notes.forEach(qn => {
        items.push({
          ...qn,
          type: 'quick_note',
          section: null
        })
      })

      // Handle search
      if (store.RECENT_NOTES_matched_page_id) {
        items = items.filter(item => item.id === store.RECENT_NOTES_matched_page_id)
      }

      // Apply filters
      if (pinnedFilter.value) {
        items = items.filter(item =>
          (item.type === 'page' && item.favorite) ||
          (item.type === 'page' && item.section && item.section.pinned)
        )
      }

      // Sort
      if (currentSort.value === 'title') {
        items.sort((a, b) => a.title.localeCompare(b.title))
      } else {
        items.sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
      }

      // Apply recency filter (limit total items)
      if (recencyFilter.value < 30) {
        const maxItems = Math.max(1, recencyFilter.value)
        items = items.slice(0, maxItems)
      }

      return items
    })

    // Actions
    const handleSearch = () => {
      // Search in sections, pages and quick_notes
      const matchInSections = dataStore.sections.find(sec =>
        sec.name.toLowerCase().includes(searchQuery.value.toLowerCase())
      )
      const matchInPages = dataStore.pages.find(pg =>
        pg.title.toLowerCase().includes(searchQuery.value.toLowerCase())
      )
      const matchInQuickNotes = dataStore.quick_notes.find(qn =>
        qn.title.toLowerCase().includes(searchQuery.value.toLowerCase())
      )

      const match = matchInSections || matchInPages || matchInQuickNotes
      if (match) {
        store.RECENT_NOTES_matched_page_id = match.id
        store.RECENT_NOTES_has_searched = true
      }
    }

    const togglePinnedFilter = () => {
      pinnedFilter.value = !pinnedFilter.value
      store.RECENT_NOTES_filters_applied = true
    }

    const handleRecencyFilter = () => {
      store.RECENT_NOTES_filters_applied = true
    }

    const handleSort = (type) => {
      currentSort.value = type
      showSortMenu.value = false
      store.RECENT_NOTES_filters_applied = true
    }

    const openItem = async (item) => {
      store.RECENT_NOTES_filters_applied = null
      store.RECENT_NOTES_has_searched = null

      store.selected_page_id = item.id
      store.note_title = item.title
      store.note_body = item.body
      store.note_tag_color = item.type === 'quick_note' ? 'yellow' : null

      store.current_page_id = 'NOTE_EDITOR'
      await router.push({ name: 'NOTE_EDITOR' })
    }

    const goHome = async () => {
      store.current_page_id = 'HOME'
      await router.push({ name: 'HOME' })
    }

    return {
      store,
      searchQuery,
      pinnedFilter,
      recencyFilter,
      showSortMenu,
      currentSort,
      hasFilters,
      hasSearched,
      filteredData,
      handleSearch,
      togglePinnedFilter,
      handleRecencyFilter,
      handleSort,
      openItem,
      goHome
    }
  }
}
</script>