<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <!-- Navbar -->
    <header class="bg-purple-700 text-white shadow-md z-20 sticky top-0">
      <div class="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
        <div class="flex items-center gap-4">
          <button 
            id="back-section-list" 
            @click="goBackSections" 
            class="hover:bg-purple-600 p-2 rounded-full transition"
          >
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
          </button>
          <h1 class="text-2xl font-bold flex items-center gap-2">
            <span>📄</span> Pages
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
            id="page-search-input"
            type="text"
            v-model="searchQuery"
            @keyup.enter="handleSearch"
            placeholder="Search pages..."
            class="w-full pl-10 pr-4 py-2 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
          />
          <svg class="w-5 h-5 text-gray-400 absolute left-3 top-2.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
        </div>

        <!-- Filters -->
        <div class="flex flex-wrap items-center gap-6">
          <!-- Favorite Checkbox -->
          <label class="flex items-center gap-2 cursor-pointer select-none">
            <div 
              id="filter-page-favorite-checkbox"
              @click="toggleFavoriteFilter"
              class="w-5 h-5 border-2 rounded flex items-center justify-center transition-colors"
              :class="favoriteFilter ? 'bg-pink-500 border-pink-500' : 'border-gray-300'"
            >
              <svg v-if="favoriteFilter" class="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
            </div>
            <span class="text-sm font-medium text-gray-700">Favorites</span>
          </label>

          <!-- Length Slider -->
          <div class="flex items-center gap-3">
            <span class="text-sm font-medium text-gray-700">Min Length: {{ lengthFilter }} chars</span>
            <input 
              id="page-length-slider"
              type="range"
              min="0"
              max="500"
              step="10"
              v-model.number="lengthFilter"
              @input="handleLengthFilter"
              class="w-32 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-purple-600"
            />
          </div>

          <!-- Sort Dropdown -->
          <div class="relative z-30">
            <button 
              id="page-sort-dropdown"
              @click="showSortMenu = !showSortMenu"
              class="flex items-center gap-2 bg-gray-100 hover:bg-gray-200 px-4 py-2 rounded-lg text-sm font-medium transition"
            >
              <span>Sort: {{ currentSort }}</span>
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
            </button>
            <div v-if="showSortMenu" class="absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-xl border border-gray-100 py-1 overflow-hidden">
              <div id="page-sort-option-recent" @click="handleSort('recent')" class="px-4 py-2 hover:bg-purple-50 cursor-pointer text-sm">Recent</div>
              <div id="page-sort-option-title" @click="handleSort('title')" class="px-4 py-2 hover:bg-purple-50 cursor-pointer text-sm">Title</div>
              <div id="page-sort-option-created" @click="handleSort('created')" class="px-4 py-2 hover:bg-purple-50 cursor-pointer text-sm">Created Date</div>
            </div>
          </div>
        </div>

        <!-- New Page Button -->
        <button 
          id="new-page-button"
          @click="createNewPage"
          class="bg-purple-600 hover:bg-purple-700 text-white font-bold py-2 px-6 rounded-lg shadow transition-colors flex items-center gap-2"
        >
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"></path></svg>
          Add Page
        </button>
      </div>

      <!-- Page List -->
      <div 
        id="page-list-container"
        class="grid grid-cols-1 gap-4"
      >
        <div 
          v-for="pg in filteredPages" 
          :key="pg.id"
          class="group bg-white rounded-xl shadow-sm hover:shadow-md transition-all duration-200 p-4 border border-gray-100 cursor-pointer flex gap-4 items-center"
          :class="{
            'page-row-filtered': hasFilters,
            'page-row-matched': hasSearched && pg.id === store.PAGE_LIST_matched_page_id,
            'page-row-visible': !hasFilters && !hasSearched
          }"
          :data-id="pg.id"
          @click="openPage(pg)"
        >
          <!-- Thumbnail -->
          <div class="w-20 h-20 bg-gray-200 rounded-lg overflow-hidden flex-shrink-0">
            <img :src="pg.image" :alt="pg.title" class="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500" />
          </div>

          <!-- Content -->
          <div class="flex-1 min-w-0">
            <div class="flex items-center gap-2 mb-1">
              <h3 class="font-bold text-gray-900 truncate group-hover:text-purple-700 transition-colors">{{ pg.title }}</h3>
              <svg v-if="pg.favorite" class="w-4 h-4 text-pink-500" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z" clip-rule="evenodd"></path></svg>
            </div>
            <p class="text-sm text-gray-500 line-clamp-2">{{ pg.body }}</p>
            <div class="text-xs text-gray-400 mt-2">
              {{ pg.created_at }} • {{ pg.length }} chars
            </div>
          </div>

          <!-- Arrow -->
          <div class="text-gray-300 group-hover:text-purple-500 transition-colors">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path></svg>
          </div>
        </div>
      </div>

      <!-- Empty State -->
      <div v-if="filteredPages.length === 0" class="text-center py-20 text-gray-500">
        <p class="text-xl">No pages found in this section.</p>
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
  name: 'PAGE_LIST',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    // UI state
    const searchQuery = ref('')
    const favoriteFilter = ref(false)
    const lengthFilter = ref(0)
    const showSortMenu = ref(false)
    const currentSort = ref(null)

    const hasFilters = computed(() => favoriteFilter.value || lengthFilter.value > 0 || currentSort.value !== 'recent')
    const hasSearched = computed(() => !!store.PAGE_LIST_matched_page_id)

    const filteredPages = computed(() => {
      // Filter by parent section
      let result = dataStore.pages.filter(pg => pg.section_id === store.selected_section_id)

      if (store.PAGE_LIST_matched_page_id) {
        return result.filter(pg => pg.id === store.PAGE_LIST_matched_page_id)
      }

      if (favoriteFilter.value) {
        result = result.filter(pg => pg.favorite)
      }

      if (lengthFilter.value > 0) {
        result = result.filter(pg => pg.length > lengthFilter.value)
      }

      if (currentSort.value === 'title') {
        result.sort((a, b) => a.title.localeCompare(b.title))
      } else if (currentSort.value === 'created') {
        result.sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
      } else {
        // recent (using created_at as proxy if updated_at not available)
        result.sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
      }

      return result
    })

    // Actions
    const handleSearch = () => {
      const match = dataStore.pages.find(pg => 
        pg.section_id === store.selected_section_id &&
        pg.title.toLowerCase().includes(searchQuery.value.toLowerCase())
      )
      if (match) {
        store.PAGE_LIST_matched_page_id = match.id
        store.PAGE_LIST_has_searched = true
      }
    }

    const toggleFavoriteFilter = () => {
      favoriteFilter.value = !favoriteFilter.value
      store.PAGE_LIST_filters_applied = true
    }

    const handleLengthFilter = () => {
      store.PAGE_LIST_filters_applied = true
    }

    const handleSort = (type) => {
      currentSort.value = type
      showSortMenu.value = false
      store.PAGE_LIST_filters_applied = true
    }

    const openPage = async (page) => {
      store.selected_page_id = page.id
      store.PAGE_LIST_filters_applied = null
      store.PAGE_LIST_has_searched = null
      
      // Load page data into store for editing
      store.note_title = page.title
      store.note_body = page.body
      store.note_tag_color = null // Reset tag color

      store.setCurrentPageId('NOTE_EDITOR')
      await router.push({ name: 'NOTE_EDITOR' })
    }

    const createNewPage = async () => {
      // Reset editor state for new page
      store.selected_page_id = null
      store.note_title = ''
      store.note_body = ''
      store.note_tag_color = null

      store.setCurrentPageId('NOTE_EDITOR')
      await router.push({ name: 'NOTE_EDITOR' })
    }

    const goBackSections = async () => {
      store.setCurrentPageId('SECTION_LIST')
      await router.push({ name: 'SECTION_LIST' })
    }

    return {
      store,
      searchQuery,
      favoriteFilter,
      lengthFilter,
      showSortMenu,
      currentSort,
      hasFilters,
      hasSearched,
      filteredPages,
      handleSearch,
      toggleFavoriteFilter,
      handleLengthFilter,
      handleSort,
      openPage,
      createNewPage,
      goBackSections
    }
  }
}
</script>