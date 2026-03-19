<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <PermissionModal />
    
    <!-- Navbar -->
    <header class="bg-purple-700 text-white shadow-md z-20">
      <div class="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
        <h1 class="text-2xl font-bold flex items-center gap-2">
          <span>📓</span> My Notebooks
        </h1>
        <div class="flex gap-4">
          <button id="back-home-button" @click="goHome" class="hover:bg-purple-600 px-3 py-1 rounded transition">Home</button>
        </div>
      </div>
    </header>

    <!-- Main Content -->
    <main class="flex-1 max-w-7xl mx-auto w-full px-4 py-8">
      
      <!-- Toolbar: Search & Filters -->
      <div class="bg-white rounded-xl shadow-sm p-4 mb-8 flex flex-col md:flex-row gap-6 items-center justify-between">
        <!-- Search -->
        <div class="relative w-full md:w-96">
          <input 
            id="notebook-search-input"
            type="text"
            v-model="searchQuery"
            @keyup.enter="handleSearch"
            placeholder="Search notebooks..."
            class="w-full pl-10 pr-4 py-2 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
          />
          <svg class="w-5 h-5 text-gray-400 absolute left-3 top-2.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
        </div>

        <!-- Filters Container -->
        <div class="flex flex-wrap items-center gap-6">
          <!-- Shared Checkbox -->
          <label class="flex items-center gap-2 cursor-pointer select-none">
            <div 
              id="filter-shared-checkbox"
              @click="toggleSharedFilter"
              class="w-5 h-5 border-2 rounded flex items-center justify-center transition-colors"
              :class="sharedFilter ? 'bg-purple-600 border-purple-600' : 'border-gray-300'"
            >
              <svg v-if="sharedFilter" class="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
            </div>
            <span class="text-sm font-medium text-gray-700">Shared Only</span>
          </label>

          <!-- Size Slider -->
          <div class="flex items-center gap-3">
            <span class="text-sm font-medium text-gray-700">Min Size: {{ sizeFilter }}MB</span>
            <input 
              id="notebook-size-slider"
              type="range"
              min="0"
              max="500"
              step="10"
              v-model.number="sizeFilter"
              @input="handleSizeFilter"
              class="w-32 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-purple-600"
            />
          </div>

          <!-- Sort Dropdown -->
          <div class="relative z-30">
            <button 
              id="notebook-sort-dropdown"
              @click="showSortMenu = !showSortMenu"
              class="flex items-center gap-2 bg-gray-100 hover:bg-gray-200 px-4 py-2 rounded-lg text-sm font-medium transition"
            >
              <span>Sort by: {{ currentSort }}</span>
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
            </button>
            <div v-if="showSortMenu" class="absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-xl border border-gray-100 py-1 overflow-hidden">
              <div id="notebook-sort-option-recent" @click="handleSort('recent')" class="px-4 py-2 hover:bg-purple-50 cursor-pointer text-sm">Recent</div>
              <div id="notebook-sort-option-name" @click="handleSort('name')" class="px-4 py-2 hover:bg-purple-50 cursor-pointer text-sm">Name</div>
              <div id="notebook-sort-option-size" @click="handleSort('size')" class="px-4 py-2 hover:bg-purple-50 cursor-pointer text-sm">Size</div>
            </div>
          </div>
        </div>

        <!-- New Notebook Button -->
        <button 
          id="new-notebook-button"
          @click="createNewNotebook"
          class="bg-purple-600 hover:bg-purple-700 text-white font-bold py-2 px-6 rounded-lg shadow transition-colors flex items-center gap-2"
        >
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"></path></svg>
          New
        </button>
      </div>

      <!-- Notebook Grid -->
      <div 
        id="notebook-list-container"
        class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
      >
        <div 
          v-for="nb in filteredNotebooks" 
          :key="nb.id"
          class="group bg-white rounded-xl shadow-sm hover:shadow-xl transition-all duration-300 border border-gray-100 overflow-hidden cursor-pointer"
          :class="{
            'notebook-row-filtered': hasFilters,
            'notebook-row-matched': hasSearched && nb.id === store.NOTEBOOK_LIST_matched_notebook_id,
            'notebook-row-visible': !hasFilters && !hasSearched
          }"
          :data-id="nb.id"
          @click="openNotebook(nb)"
        >
          <!-- Cover Image -->
          <div class="h-40 bg-gray-200 relative overflow-hidden">
            <img :src="nb.image" :alt="nb.name" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" />
            <div class="absolute top-2 right-2 bg-black/50 text-white text-xs px-2 py-1 rounded backdrop-blur-sm">
              {{ nb.size }}MB
            </div>
          </div>
          
          <!-- Content -->
          <div class="p-5">
            <h3 class="font-bold text-lg text-gray-900 mb-1 group-hover:text-purple-700 transition-colors">{{ nb.name }}</h3>
            <p class="text-sm text-gray-500 mb-4">Last edited: {{ nb.created_at }}</p>
            <div class="flex items-center justify-between text-xs text-gray-400 border-t pt-3">
              <span v-if="nb.shared" class="flex items-center gap-1 text-blue-500 bg-blue-50 px-2 py-0.5 rounded-full">
                <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z"></path></svg>
                Shared
              </span>
              <span v-else>Private</span>
              <span>ID: {{ nb.id }}</span>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Empty State -->
      <div v-if="filteredNotebooks.length === 0" class="text-center py-20 text-gray-500">
        <p class="text-xl">No notebooks found.</p>
      </div>

    </main>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'
import PermissionModal from '../components/PermissionModal.vue'

export default {
  name: 'NOTEBOOK_LIST',
  components: { PermissionModal },
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    // Local state for UI controls
    const searchQuery = ref('')
    const sharedFilter = ref(false)
    const sizeFilter = ref(0)
    const showSortMenu = ref(false)
    const currentSort = ref(null)

    // Computed properties for filtering logic
    const hasFilters = computed(() => sharedFilter.value || sizeFilter.value > 0 || currentSort.value !== 'recent')
    const hasSearched = computed(() => !!store.NOTEBOOK_LIST_matched_notebook_id)

    const filteredNotebooks = computed(() => {
      let result = [...dataStore.notebooks]

      // Search match first if exists
      if (store.NOTEBOOK_LIST_matched_notebook_id) {
        return result.filter(nb => nb.id === store.NOTEBOOK_LIST_matched_notebook_id)
      }

      // Filter: Shared
      if (sharedFilter.value) {
        result = result.filter(nb => nb.shared)
      }

      // Filter: Size (Greater than slider value)
      if (sizeFilter.value > 0) {
        result = result.filter(nb => nb.size > sizeFilter.value)
      }

      // Sort
      if (currentSort.value === 'name') {
        result.sort((a, b) => a.name.localeCompare(b.name))
      } else if (currentSort.value === 'size') {
        result.sort((a, b) => b.size - a.size) // Descending size
      } else {
        // recent (default)
        result.sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
      }

      return result
    })

    // Actions
    const handleSearch = () => {
      const match = dataStore.notebooks.find(nb => 
        nb.name.toLowerCase().includes(searchQuery.value.toLowerCase())
      )
      if (match) {
        store.NOTEBOOK_LIST_matched_notebook_id = match.id
        store.NOTEBOOK_LIST_has_searched = true
      }
    }

    const toggleSharedFilter = () => {
      sharedFilter.value = !sharedFilter.value
      store.NOTEBOOK_LIST_filters_applied = true
    }

    const handleSizeFilter = () => {
      store.NOTEBOOK_LIST_filters_applied = true
    }

    const handleSort = (type) => {
      currentSort.value = type
      showSortMenu.value = false
      store.NOTEBOOK_LIST_filters_applied = true
    }

    const openNotebook = async (notebook) => {
      store.selected_notebook_id = notebook.id
      // Clear temporary states
      store.NOTEBOOK_LIST_filters_applied = null
      store.NOTEBOOK_LIST_has_searched = null
      
      store.setCurrentPageId('SECTION_LIST')
      await router.push({ name: 'SECTION_LIST' })
    }

    const createNewNotebook = async () => {
      store.setCurrentPageId('NOTEBOOK_CREATE')
      await router.push({ name: 'NOTEBOOK_CREATE' })
    }

    const goHome = async () => {
      store.setCurrentPageId('HOME')
      await router.push({ name: 'HOME' })
    }

    return {
      store,
      searchQuery,
      sharedFilter,
      sizeFilter,
      showSortMenu,
      currentSort,
      hasFilters,
      hasSearched,
      filteredNotebooks,
      handleSearch,
      toggleSharedFilter,
      handleSizeFilter,
      handleSort,
      openNotebook,
      createNewNotebook,
      goHome
    }
  }
}
</script>