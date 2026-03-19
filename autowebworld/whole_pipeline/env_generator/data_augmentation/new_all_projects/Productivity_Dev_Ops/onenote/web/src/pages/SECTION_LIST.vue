<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <!-- Navbar -->
    <header class="bg-purple-700 text-white shadow-md z-20 sticky top-0">
      <div class="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
        <div class="flex items-center gap-4">
          <button 
            id="back-notebooks" 
            @click="goBackNotebooks" 
            class="hover:bg-purple-600 p-2 rounded-full transition"
          >
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
          </button>
          <h1 class="text-2xl font-bold flex items-center gap-2">
            <span>📑</span> Sections
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
            id="section-search-input"
            type="text"
            v-model="searchQuery"
            @keyup.enter="handleSearch"
            placeholder="Search sections..."
            class="w-full pl-10 pr-4 py-2 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
          />
          <svg class="w-5 h-5 text-gray-400 absolute left-3 top-2.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
        </div>

        <!-- Filters -->
        <div class="flex flex-wrap items-center gap-6">
          <!-- Pinned Checkbox -->
          <label class="flex items-center gap-2 cursor-pointer select-none">
            <div 
              id="filter-section-pinned-checkbox"
              @click="togglePinnedFilter"
              class="w-5 h-5 border-2 rounded flex items-center justify-center transition-colors"
              :class="pinnedFilter ? 'bg-yellow-500 border-yellow-500' : 'border-gray-300'"
            >
              <svg v-if="pinnedFilter" class="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
            </div>
            <span class="text-sm font-medium text-gray-700">Pinned Only</span>
          </label>

          <!-- Activity Slider -->
          <div class="flex items-center gap-3">
            <span class="text-sm font-medium text-gray-700">Min Activity: {{ activityFilter }}%</span>
            <input 
              id="section-activity-slider"
              type="range"
              min="0"
              max="100"
              step="5"
              v-model.number="activityFilter"
              @input="handleActivityFilter"
              class="w-32 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-purple-600"
            />
          </div>

          <!-- Sort Dropdown -->
          <div class="relative z-30">
            <button 
              id="section-sort-dropdown"
              @click="showSortMenu = !showSortMenu"
              class="flex items-center gap-2 bg-gray-100 hover:bg-gray-200 px-4 py-2 rounded-lg text-sm font-medium transition"
            >
              <span>Sort: {{ currentSort }}</span>
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
            </button>
            <div v-if="showSortMenu" class="absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-xl border border-gray-100 py-1 overflow-hidden">
              <div id="section-sort-option-recent" @click="handleSort('recent')" class="px-4 py-2 hover:bg-purple-50 cursor-pointer text-sm">Recent</div>
              <div id="section-sort-option-name" @click="handleSort('name')" class="px-4 py-2 hover:bg-purple-50 cursor-pointer text-sm">Name</div>
              <div id="section-sort-option-custom" @click="handleSort('custom')" class="px-4 py-2 hover:bg-purple-50 cursor-pointer text-sm">Custom</div>
            </div>
          </div>
        </div>

        <!-- New Section Button -->
        <button 
          id="new-section-button"
          @click="createNewSection"
          class="bg-purple-600 hover:bg-purple-700 text-white font-bold py-2 px-6 rounded-lg shadow transition-colors flex items-center gap-2"
        >
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"></path></svg>
          Add Section
        </button>
      </div>

      <!-- Section List -->
      <div 
        id="section-list-container"
        class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
      >
        <div 
          v-for="sec in filteredSections" 
          :key="sec.id"
          class="group bg-white rounded-xl shadow-sm hover:shadow-lg transition-all duration-300 border-l-4 overflow-hidden cursor-pointer flex flex-col h-full"
          :class="{
            'border-purple-500': !sec.pinned,
            'border-yellow-500': sec.pinned,
            'section-row-filtered': hasFilters,
            'section-row-matched': hasSearched && sec.id === store.SECTION_LIST_matched_section_id,
            'section-row-visible': !hasFilters && !hasSearched
          }"
          :data-id="sec.id"
          @click="openSection(sec)"
        >
          <!-- Thumbnail -->
          <div class="h-32 bg-gray-100 overflow-hidden relative">
             <img :src="sec.image" :alt="sec.name" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" />
             <div v-if="sec.pinned" class="absolute top-2 right-2 text-yellow-500 bg-white rounded-full p-1 shadow-md">
               <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20"><path d="M5 4a2 2 0 012-2h6a2 2 0 012 2v14l-5-2.5L5 18V4z"></path></svg>
             </div>
          </div>

          <!-- Info -->
          <div class="p-4 flex-1 flex flex-col">
            <h3 class="font-bold text-gray-900 group-hover:text-purple-700 transition-colors">{{ sec.name }}</h3>
            <div class="mt-auto pt-4 flex items-center justify-between text-xs text-gray-500">
              <span class="flex items-center gap-1">
                <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
                {{ sec.activity }}%
              </span>
              <span>{{ sec.created_at }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Empty State -->
      <div v-if="filteredSections.length === 0" class="text-center py-20 text-gray-500">
        <p class="text-xl">No sections found in this notebook.</p>
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
  name: 'SECTION_LIST',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    // UI state
    const searchQuery = ref('')
    const pinnedFilter = ref(false)
    const activityFilter = ref(0)
    const showSortMenu = ref(false)
    const currentSort = ref(null)

    const hasFilters = computed(() => pinnedFilter.value || activityFilter.value > 0 || currentSort.value !== 'recent')
    const hasSearched = computed(() => !!store.SECTION_LIST_matched_section_id)

    const filteredSections = computed(() => {
      // Filter by parent notebook first
      let result = dataStore.sections.filter(sec => sec.notebook_id === store.selected_notebook_id)

      if (store.SECTION_LIST_matched_section_id) {
        return result.filter(sec => sec.id === store.SECTION_LIST_matched_section_id)
      }

      if (pinnedFilter.value) {
        result = result.filter(sec => sec.pinned)
      }

      if (activityFilter.value > 0) {
        result = result.filter(sec => sec.activity > activityFilter.value)
      }

      if (currentSort.value === 'name') {
        result.sort((a, b) => a.name.localeCompare(b.name))
      } else if (currentSort.value === 'custom') {
        // Just a dummy sort for FSM compliance, maybe by ID
        result.sort((a, b) => a.id.localeCompare(b.id))
      } else {
        // recent
        result.sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
      }

      return result
    })

    // Actions
    const handleSearch = () => {
      const match = dataStore.sections.find(sec => 
        sec.notebook_id === store.selected_notebook_id &&
        sec.name.toLowerCase().includes(searchQuery.value.toLowerCase())
      )
      if (match) {
        store.SECTION_LIST_matched_section_id = match.id
        store.SECTION_LIST_has_searched = true
      }
    }

    const togglePinnedFilter = () => {
      pinnedFilter.value = !pinnedFilter.value
      store.SECTION_LIST_filters_applied = true
    }

    const handleActivityFilter = () => {
      store.SECTION_LIST_filters_applied = true
    }

    const handleSort = (type) => {
      currentSort.value = type
      showSortMenu.value = false
      store.SECTION_LIST_filters_applied = true
    }

    const openSection = async (section) => {
      store.selected_section_id = section.id
      store.SECTION_LIST_filters_applied = null
      store.SECTION_LIST_has_searched = null
      
      store.setCurrentPageId('PAGE_LIST')
      await router.push({ name: 'PAGE_LIST' })
    }

    const createNewSection = async () => {
      store.setCurrentPageId('SECTION_CREATE')
      await router.push({ name: 'SECTION_CREATE' })
    }

    const goBackNotebooks = async () => {
      store.setCurrentPageId('NOTEBOOK_LIST')
      await router.push({ name: 'NOTEBOOK_LIST' })
    }

    return {
      store,
      searchQuery,
      pinnedFilter,
      activityFilter,
      showSortMenu,
      currentSort,
      hasFilters,
      hasSearched,
      filteredSections,
      handleSearch,
      togglePinnedFilter,
      handleActivityFilter,
      handleSort,
      openSection,
      createNewSection,
      goBackNotebooks
    }
  }
}
</script>