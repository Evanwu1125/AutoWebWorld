<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <header class="bg-white shadow-sm z-10">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div class="flex items-center">
          <button id="logo-home" @click="goHome" class="text-2xl font-bold text-blue-600 mr-8">Optimizely</button>
          <h1 class="text-xl font-semibold text-gray-800">Audiences</h1>
        </div>
        <button 
          id="btn-new-audience"
          @click="createAudience"
          class="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700"
        >
          New Audience
        </button>
      </div>
    </header>

    <main class="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
      <!-- Controls -->
      <div class="bg-white p-4 rounded-lg shadow-sm mb-6 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div class="relative flex-1 max-w-lg">
          <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <svg class="h-5 w-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>
          <input 
            id="audiences-search-input"
            v-model="searchQuery"
            @keyup.enter="performSearch"
            type="text" 
            class="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md leading-5 bg-white placeholder-gray-500 focus:outline-none focus:placeholder-gray-400 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 sm:text-sm" 
            placeholder="Search audiences..."
          >
        </div>

        <div class="flex items-center gap-4">
          <div class="flex items-center">
            <input 
              id="audiences-filter-saved-checkbox" 
              type="checkbox" 
              v-model="filterSaved"
              @change="applyFilters"
              class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
            >
            <label for="audiences-filter-saved-checkbox" class="ml-2 block text-sm text-gray-700">
              Saved Only
            </label>
          </div>

          <div class="w-48">
            <input 
              id="audiences-size-slider"
              type="range" 
              v-model="sizeThreshold"
              @input="applyFilters"
              min="0"
              max="100000"
              step="5000"
              class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
            >
            <div class="text-xs text-gray-500 text-center mt-1">Size > {{ sizeThreshold }}</div>
          </div>

          <div class="relative" id="audiences-sort-dropdown">
            <button @click="toggleSort" class="bg-white border border-gray-300 text-gray-700 px-4 py-2 rounded-md text-sm font-medium hover:bg-gray-50 flex items-center shadow-sm">
              Sort
              <svg class="ml-2 h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            <div v-if="sortOpen" class="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg z-50 ring-1 ring-black ring-opacity-5">
              <div class="py-1">
                <div id="audiences-sort-option-size" @click="sort('size')" class="cursor-pointer block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100">Size</div>
                <div id="audiences-sort-option-name" @click="sort('name')" class="cursor-pointer block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100">Name</div>
                <div id="audiences-sort-option-last-modified" @click="sort('last_modified')" class="cursor-pointer block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100">Last Modified</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Table -->
      <div id="audiences-table" class="bg-white shadow-sm rounded-lg overflow-hidden">
        <ul class="divide-y divide-gray-200">
          <li v-for="audience in filteredAudiences" :key="audience.id" class="hover:bg-gray-50">
            <div 
              :class="[
                'px-6 py-4 flex items-center cursor-pointer',
                `data-id-${audience.id}`,
                isSearched ? 'row-matched' : '',
                isFiltered ? 'row-filtered' : 'row-visible'
              ]"
              @click="openAudience(audience)"
            >
              <div class="flex-shrink-0 h-12 w-12">
                 <img :src="audience.image" class="h-12 w-12 rounded-md object-cover" alt="" />
              </div>
              <div class="ml-4 flex-1">
                <div class="flex items-center justify-between">
                  <h3 class="text-sm font-medium text-gray-900">{{ audience.name }}</h3>
                  <p class="text-sm text-gray-500">{{ audience.size.toLocaleString() }} users</p>
                </div>
                <div class="flex items-center justify-between mt-1">
                  <p class="text-sm text-gray-500">{{ audience.description }}</p>
                  <p class="text-xs text-gray-400">{{ audience.last_modified }}</p>
                </div>
              </div>
              <div class="ml-4">
                 <svg class="h-5 w-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                   <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
                 </svg>
              </div>
            </div>
          </li>
        </ul>
        <div v-if="filteredAudiences.length === 0" class="p-8 text-center text-gray-500">
          No audiences found.
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
  name: 'AUDIENCES_LIST',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const searchQuery = ref('')
    const filterSaved = ref(false)
    const sizeThreshold = ref(0)
    const sortOpen = ref(false)
    const activeSort = ref(null)
    
    const isSearched = ref(false)
    const isFiltered = ref(false)

    function performSearch() {
      isSearched.value = true
      signatureStore.audiences_list_has_searched = true
      signatureStore.audiences_list_matched_item_id = searchQuery.value
    }

    function applyFilters() {
      isFiltered.value = true
      signatureStore.audiences_list_filters_applied = true
    }

    function toggleSort() {
      sortOpen.value = !sortOpen.value
    }

    function sort(field) {
      activeSort.value = field
      sortOpen.value = false
      applyFilters()
    }

    const filteredAudiences = computed(() => {
      let items = [...dataStore.audiences]

      if (searchQuery.value) {
        const q = searchQuery.value.toLowerCase()
        items = items.filter(a => a.name.toLowerCase().includes(q) || a.description.toLowerCase().includes(q))
      }

      if (sizeThreshold.value > 0) {
        items = items.filter(a => a.size > sizeThreshold.value)
      }

      // Mock logic for "Saved Only" - assume all are saved in mock
      // but maybe filter by a property if available. 
      
      if (activeSort.value) {
        items.sort((a, b) => {
          if (activeSort.value === 'size') return b.size - a.size
          if (activeSort.value === 'name') return a.name.localeCompare(b.name)
          if (activeSort.value === 'last_modified') return new Date(b.last_modified) - new Date(a.last_modified)
          return 0
        })
      }

      return items
    })

    function openAudience(audience) {
      if (isSearched.value) {
        signatureStore.audiences_list_matched_item_id = audience.id
      } else if (isFiltered.value) {
        signatureStore.audiences_list_filters_applied = true
      } else {
        signatureStore.audiences_list_viewport_anchor_id = audience.id
      }
      
      signatureStore.audiences_list_selected_item_id = audience.id
      signatureStore.setCurrentPageId('AUDIENCE_DETAIL')
      router.push({ name: 'AUDIENCE_DETAIL', params: { id: audience.id } })
    }

    function createAudience() {
      signatureStore.setCurrentPageId('AUDIENCE_CREATE')
      router.push({ name: 'AUDIENCE_CREATE' })
    }

    function goHome() {
      signatureStore.setCurrentPageId('HOME')
      router.push({ name: 'HOME' })
    }

    return {
      searchQuery,
      filterSaved,
      sizeThreshold,
      sortOpen,
      filteredAudiences,
      isSearched,
      isFiltered,
      performSearch,
      applyFilters,
      toggleSort,
      sort,
      openAudience,
      createAudience,
      goHome
    }
  }
}
</script>