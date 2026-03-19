<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <!-- Header -->
    <header class="bg-white shadow-sm z-10">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div class="flex items-center">
          <button id="logo-home" @click="goHome" class="text-2xl font-bold text-blue-600 mr-8">Optimizely</button>
          <h1 class="text-xl font-semibold text-gray-800">Experiments</h1>
        </div>
        <button 
          id="btn-new-experiment"
          @click="createExperiment"
          class="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700"
        >
          New Experiment
        </button>
      </div>
    </header>

    <main class="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
      <!-- Controls -->
      <div class="bg-white p-4 rounded-lg shadow-sm mb-6 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <!-- Search -->
        <div class="relative flex-1 max-w-lg">
          <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <svg class="h-5 w-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>
          <input 
            id="experiments-search-input"
            v-model="searchQuery"
            @keyup.enter="performSearch"
            type="text" 
            class="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md leading-5 bg-white placeholder-gray-500 focus:outline-none focus:placeholder-gray-400 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 sm:text-sm" 
            placeholder="Search experiments..."
          >
        </div>

        <div class="flex items-center gap-4">
          <!-- Filter Checkbox -->
          <div class="flex items-center">
            <input 
              id="experiments-filter-running-checkbox" 
              type="checkbox" 
              v-model="filterRunning"
              @change="applyFilters"
              class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
            >
            <label for="experiments-filter-running-checkbox" class="ml-2 block text-sm text-gray-700">
              Running Only
            </label>
          </div>

          <!-- Filter Slider -->
          <div class="w-48">
            <input 
              id="experiments-type-slider"
              type="range" 
              v-model="visitorThreshold"
              @input="applyFilters"
              min="0"
              max="50000"
              step="1000"
              class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
            >
            <div class="text-xs text-gray-500 text-center mt-1">Visitors > {{ visitorThreshold }}</div>
          </div>

          <!-- Sort Dropdown -->
          <div class="relative" id="experiments-sort-dropdown">
            <button @click="toggleSort" class="bg-white border border-gray-300 text-gray-700 px-4 py-2 rounded-md text-sm font-medium hover:bg-gray-50 flex items-center shadow-sm">
              Sort
              <svg class="ml-2 h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            <div v-if="sortOpen" class="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg z-50 ring-1 ring-black ring-opacity-5">
              <div class="py-1">
                <div id="experiments-sort-option-last-modified-desc" @click="sort('last_modified')" class="cursor-pointer block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100">Last Modified</div>
                <div id="experiments-sort-option-created" @click="sort('created')" class="cursor-pointer block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100">Date Created</div>
                <div id="experiments-sort-option-status" @click="sort('status')" class="cursor-pointer block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100">Status</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Grid/Table -->
      <div id="experiments-table" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div 
          v-for="exp in filteredExperiments" 
          :key="exp.id"
          :class="[
            'bg-white rounded-lg shadow-sm overflow-hidden hover:shadow-md transition-shadow cursor-pointer group flex flex-col',
            `data-id-${exp.id}`,
            isSearched ? 'row-matched' : '',
            isFiltered ? 'row-filtered' : 'row-visible'
          ]"
          @click="openExperiment(exp)"
        >
          <!-- Image Thumbnail -->
          <div class="h-40 w-full relative bg-gray-200">
             <img :src="exp.image" class="w-full h-full object-cover group-hover:opacity-90 transition-opacity" alt="Experiment Thumbnail" />
             <div class="absolute top-2 right-2">
               <span :class="[
                 'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium',
                 exp.status === 'Running' ? 'bg-green-100 text-green-800' : 
                 exp.status === 'Paused' ? 'bg-yellow-100 text-yellow-800' : 'bg-gray-100 text-gray-800'
               ]">
                 {{ exp.status }}
               </span>
             </div>
          </div>
          
          <!-- Content -->
          <div class="p-4 flex-1 flex flex-col">
            <h3 class="text-lg font-medium text-gray-900 mb-1 group-hover:text-blue-600">{{ exp.name }}</h3>
            <p class="text-sm text-gray-500 mb-4">{{ exp.type }}</p>
            
            <div class="mt-auto grid grid-cols-2 gap-4 text-sm">
              <div>
                <div class="text-gray-500">Visitors</div>
                <div class="font-semibold">{{ exp.visitors.toLocaleString() }}</div>
              </div>
              <div>
                <div class="text-gray-500">Conversions</div>
                <div class="font-semibold">{{ exp.conversions }}</div>
              </div>
            </div>
            
            <div class="mt-4 pt-4 border-t border-gray-100 text-xs text-gray-400 flex justify-between">
               <span>Created: {{ exp.created }}</span>
               <span>Modified: {{ exp.last_modified }}</span>
            </div>
          </div>
        </div>
      </div>
      
      <div v-if="filteredExperiments.length === 0" class="text-center py-12">
        <svg class="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
        </svg>
        <h3 class="mt-2 text-sm font-medium text-gray-900">No experiments found</h3>
        <p class="mt-1 text-sm text-gray-500">Try adjusting your search or filters.</p>
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
  name: 'EXPERIMENTS_LIST',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const searchQuery = ref('')
    const filterRunning = ref(false)
    const visitorThreshold = ref(0)
    const sortOpen = ref(false)
    const activeSort = ref(null)
    
    // State flags for FSM logic
    const isSearched = ref(false)
    const isFiltered = ref(false)

    function performSearch() {
      isSearched.value = true
      signatureStore.experiments_list_has_searched = true
      signatureStore.experiments_list_matched_item_id = searchQuery.value // Store the query or first matched ID
    }

    function applyFilters() {
      isFiltered.value = true
      signatureStore.experiments_list_filters_applied = true
    }

    function toggleSort() {
      sortOpen.value = !sortOpen.value
    }

    function sort(field) {
      activeSort.value = field
      sortOpen.value = false
      applyFilters()
    }

    const filteredExperiments = computed(() => {
      let items = [...dataStore.experiments]

      if (searchQuery.value) {
        const q = searchQuery.value.toLowerCase()
        items = items.filter(e => e.name.toLowerCase().includes(q))
      }

      if (filterRunning.value) {
        items = items.filter(e => e.status === 'Running')
      }

      if (visitorThreshold.value > 0) {
        items = items.filter(e => e.visitors > visitorThreshold.value)
      }

      if (activeSort.value) {
        items.sort((a, b) => {
          if (activeSort.value === 'visitors') return b.visitors - a.visitors
          if (activeSort.value === 'created') return new Date(b.created) - new Date(a.created)
          if (activeSort.value === 'last_modified') return new Date(b.last_modified) - new Date(a.last_modified)
          return a.status.localeCompare(b.status)
        })
      }

      return items
    })

    function openExperiment(exp) {
      if (isSearched.value) {
        signatureStore.experiments_list_matched_item_id = exp.id
      } else if (isFiltered.value) {
        signatureStore.experiments_list_filters_applied = true // Reset just in case
      } else {
        signatureStore.experiments_list_viewport_anchor_id = exp.id // For scroll logic
      }
      
      signatureStore.experiments_list_selected_item_id = exp.id
      signatureStore.setCurrentPageId('EXPERIMENT_DETAIL')
      router.push({ name: 'EXPERIMENT_DETAIL', params: { id: exp.id } })
    }

    function createExperiment() {
      signatureStore.setCurrentPageId('EXPERIMENT_CREATE_TYPE')
      router.push({ name: 'EXPERIMENT_CREATE_TYPE' })
    }

    function goHome() {
      signatureStore.setCurrentPageId('HOME')
      router.push({ name: 'HOME' })
    }

    return {
      searchQuery,
      filterRunning,
      visitorThreshold,
      sortOpen,
      filteredExperiments,
      isSearched,
      isFiltered,
      performSearch,
      applyFilters,
      toggleSort,
      sort,
      openExperiment,
      createExperiment,
      goHome
    }
  }
}
</script>