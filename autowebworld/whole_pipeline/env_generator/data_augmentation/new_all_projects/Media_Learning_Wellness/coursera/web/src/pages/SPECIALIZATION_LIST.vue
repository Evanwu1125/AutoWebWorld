<template>
  <div class="min-h-screen bg-gray-50">
    <!-- Nav -->
    <nav class="bg-white shadow-sm sticky top-0 z-20">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between h-16">
          <div class="flex items-center">
            <div id="header-logo-home" class="flex-shrink-0 flex items-center cursor-pointer" @click="goHome">
              <span class="text-2xl font-bold text-blue-700">Coursera</span>
            </div>
            <div class="ml-10 flex items-baseline space-x-4">
               <h1 class="text-xl font-semibold text-gray-800">Specializations</h1>
            </div>
          </div>
        </div>
      </div>
    </nav>

    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <!-- Search Bar -->
      <div class="flex justify-center mb-8">
        <div class="w-full max-w-2xl relative">
          <input 
            id="specialization-search-input"
            type="text" 
            v-model="searchQuery"
            @keypress.enter="handleSearch"
            placeholder="Search specializations" 
            class="w-full px-5 py-3 border border-gray-300 rounded-full shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent text-lg"
          >
          <button 
            @click="handleSearch"
            class="absolute right-2 top-2 bg-blue-700 text-white p-2 rounded-full hover:bg-blue-800"
          >
            <svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </button>
        </div>
      </div>

      <div class="flex flex-col lg:flex-row gap-8">
        <!-- Sidebar Filters -->
        <div class="w-full lg:w-64 flex-shrink-0 space-y-6">
          <div class="bg-white p-6 rounded-lg shadow-sm border border-gray-100">
            <h3 class="text-lg font-medium text-gray-900 mb-4">Filters</h3>
            
            <div class="mb-6">
              <h4 class="text-sm font-semibold text-gray-700 mb-2">Level</h4>
              <div class="flex items-center">
                <input 
                  id="filter-beginner-checkbox"
                  type="checkbox" 
                  v-model="filterBeginner"
                  @change="applyFilters"
                  class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                >
                <label for="filter-beginner-checkbox" class="ml-2 block text-sm text-gray-700">
                  Beginner
                </label>
              </div>
            </div>

            <div>
              <h4 class="text-sm font-semibold text-gray-700 mb-2">Duration (Months)</h4>
               <div class="flex items-center justify-between text-xs text-gray-500 mb-1">
                <span>{{ minDuration }}</span>
                <span>{{ maxDuration }}+</span>
              </div>
              <input 
                id="filter-duration-slider"
                type="range" 
                :min="minDuration" 
                :max="maxDuration"
                step="1"
                v-model="durationFilterValue"
                @input="applyFilters"
                class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              >
              <div class="text-center text-sm font-medium text-blue-700 mt-2">
                Min: {{ durationFilterValue }} months
              </div>
            </div>
          </div>
        </div>

        <!-- Main Content -->
        <div class="flex-1">
          <!-- Sort -->
          <div class="flex justify-end mb-6">
            <div class="relative">
              <button 
                id="spec-sort-dropdown"
                @click="toggleSortMenu"
                class="inline-flex justify-center w-full rounded-md border border-gray-300 shadow-sm px-4 py-2 bg-white text-sm font-medium text-gray-700 hover:bg-gray-50 focus:outline-none"
              >
                Sort by: {{ currentSortLabel }}
                <svg class="ml-2 -mr-1 h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
                </svg>
              </button>

              <div v-if="isSortMenuOpen" class="origin-top-right absolute right-0 mt-2 w-40 rounded-md shadow-lg bg-white ring-1 ring-black ring-opacity-5 z-10">
                <div class="py-1" role="menu">
                  <div 
                    id="spec-sort-option-popular"
                    @click="setSort('popular', 'Most Popular')"
                    class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer"
                  >
                    Most Popular
                  </div>
                  <div 
                    id="spec-sort-option-newest"
                    @click="setSort('newest', 'Newest')"
                    class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer"
                  >
                    Newest
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- List -->
          <div id="specialization-results-list" class="space-y-6">
            <div id="specialization-results">
            <div
              v-for="spec in filteredSpecs"
              :key="spec.id"
              :class="[getCardClass(spec), `data-id-${spec.id}`]"
              class="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden hover:shadow-md transition-shadow cursor-pointer flex flex-col sm:flex-row"
              @click="openSpec(spec)"
            >
              <div class="sm:w-48 h-48 sm:h-auto flex-shrink-0 bg-gray-200">
                <img :src="spec.image" :alt="spec.title" class="w-full h-full object-cover">
              </div>
              <div class="p-6 flex-1 flex flex-col justify-between">
                <div>
                  <div class="flex items-center text-xs font-semibold tracking-wide uppercase text-blue-600 mb-1">
                    {{ spec.university }}
                  </div>
                  <h3 class="text-xl font-bold text-gray-900 mb-2">{{ spec.title }}</h3>
                  <p class="text-gray-600 text-sm mb-4 line-clamp-2">{{ spec.description }}</p>
                  
                  <div class="flex items-center space-x-4 text-sm text-gray-500">
                    <span class="flex items-center">
                      <span class="text-yellow-400 mr-1">★</span> {{ spec.rating }}
                    </span>
                    <span>{{ spec.courses_count }} Courses</span>
                    <span>{{ spec.level }}</span>
                  </div>
                </div>
                
                <div class="mt-4 flex items-center justify-between">
                   <span class="text-sm font-medium text-gray-900">{{ spec.duration }} months</span>
                </div>
              </div>
            </div>
            </div>

            <div v-if="filteredSpecs.length === 0" class="text-center py-12">
               <p class="text-gray-500 text-lg">No specializations found.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'SPECIALIZATION_LIST',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const searchQuery = ref('')
    const filterBeginner = ref(false)
    const durationFilterValue = ref(0)
    const currentSort = ref(null)
    const currentSortLabel = ref('Default')
    const isSortMenuOpen = ref(false)

    const minDuration = computed(() => {
      const durations = dataStore.specializations.map(c => c.duration)
      return Math.min(...durations)
    })
    
    const maxDuration = computed(() => {
      const durations = dataStore.specializations.map(c => c.duration)
      return Math.max(...durations)
    })
    
    durationFilterValue.value = minDuration.value

    const filteredSpecs = computed(() => {
      let result = [...dataStore.specializations]

      if (filterBeginner.value) {
        result = result.filter(s => s.level.toLowerCase() === 'beginner')
      }

      if (durationFilterValue.value > minDuration.value) {
        result = result.filter(s => s.duration >= durationFilterValue.value)
      }

      if (searchQuery.value) {
         const q = searchQuery.value.toLowerCase()
         result = result.filter(s => 
           s.title.toLowerCase().includes(q) || 
           s.university.toLowerCase().includes(q)
         )
      }

      if (currentSort.value === 'newest') {
        result.sort((a, b) => b.id.localeCompare(a.id))
      }

      return result
    })

    function applyFilters() {
      store.specialization_list_filters_applied = true
    }

    function handleSearch() {
      store.specialization_list_has_searched = true
      const match = filteredSpecs.value.length > 0 ? filteredSpecs.value[0] : null
      store.matched_specialization_id = match ? match.id : null
    }

    function toggleSortMenu() {
      isSortMenuOpen.value = !isSortMenuOpen.value
    }

    function setSort(value, label) {
      currentSort.value = value
      currentSortLabel.value = label
      isSortMenuOpen.value = false
    }

    function getCardClass(spec) {
      if (store.specialization_list_has_searched && spec.id === store.matched_specialization_id) {
        return 'specialization-card-matched'
      }
      if (store.specialization_list_filters_applied) {
        return 'specialization-card-filtered'
      }
      return 'specialization-card-visible'
    }

    async function openSpec(spec) {
      store.selected_specialization_id = spec.id
      store.setCurrentPageId('SPECIALIZATION_DETAIL')
      await router.push({ name: 'SPECIALIZATION_DETAIL', params: { id: spec.id } })
    }

    async function goHome() {
      store.setCurrentPageId('HOME')
      await router.push({ name: 'HOME' })
    }

    return {
      store,
      searchQuery,
      filterBeginner,
      durationFilterValue,
      minDuration,
      maxDuration,
      filteredSpecs,
      currentSortLabel,
      isSortMenuOpen,
      applyFilters,
      handleSearch,
      toggleSortMenu,
      setSort,
      getCardClass,
      openSpec,
      goHome
    }
  }
}
</script>