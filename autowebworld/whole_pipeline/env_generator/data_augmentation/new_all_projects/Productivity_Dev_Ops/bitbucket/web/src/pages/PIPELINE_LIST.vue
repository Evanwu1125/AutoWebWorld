<template>
  <div class="min-h-screen bg-[#FAFBFC] flex flex-col">
    <!-- Header -->
    <header class="bg-white border-b border-gray-200 px-6 py-4 flex justify-between items-center sticky top-0 z-20">
      <div class="flex items-center space-x-4">
        <button id="pipeline-list-back-home" @click="goHome" class="text-gray-500 hover:text-blue-600 transition-colors">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"/></svg>
        </button>
        <h1 class="text-2xl font-bold text-[#172B4D]">Pipelines</h1>
      </div>
      <button 
        id="configure-pipeline-button" 
        @click="goToConfigure"
        class="bg-[#0052CC] text-white px-4 py-2 rounded-md font-medium hover:bg-blue-700 transition-colors shadow-sm flex items-center"
      >
        <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"/></svg>
        Configure pipeline
      </button>
    </header>

    <div class="flex flex-1 container mx-auto px-6 py-8 gap-8">
      <!-- Sidebar Filters -->
      <aside class="w-64 flex-shrink-0 space-y-6">
        <div class="bg-white p-4 rounded-lg shadow-sm border border-gray-200">
          <h3 class="font-bold text-[#172B4D] mb-4 uppercase text-xs tracking-wider">Filters</h3>
          
          <!-- Search -->
          <div class="mb-6">
            <div class="relative">
              <input 
                id="pipeline-search-input"
                v-model="searchQuery"
                @keyup.enter="handleSearch"
                type="text" 
                placeholder="Find a pipeline..."
                class="w-full pl-9 pr-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
              <svg class="w-4 h-4 text-gray-400 absolute left-3 top-2.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
            </div>
          </div>

          <!-- Checkboxes -->
          <div class="space-y-3 mb-6">
            <label class="flex items-center space-x-2 cursor-pointer text-sm text-gray-700 hover:text-blue-600">
              <input type="checkbox" id="filter-failed-pipelines-checkbox" v-model="filterFailed" class="form-checkbox text-blue-600 rounded">
              <span>Failed only</span>
            </label>
          </div>

          <!-- Slider: Recent (Date) -->
          <div class="mb-6">
            <label class="block text-sm font-medium text-gray-700 mb-2">
              Run After: {{ new Date(filterDate).toLocaleDateString() }}
            </label>
            <input 
              id="pipeline-recent-slider"
              type="range" 
              v-model.number="filterDate" 
              :min="minDate" 
              :max="maxDate"
              step="86400000"
              class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
            >
          </div>
        </div>
      </aside>

      <!-- Main Content -->
      <div class="flex-1">
        <!-- Toolbar -->
        <div class="flex justify-between items-center mb-6">
          <div class="text-sm text-gray-500">
            Showing <span class="font-bold text-gray-900">{{ filteredPipelines.length }}</span> pipelines
          </div>
          
          <!-- Sort Dropdown -->
          <div class="relative group" id="pipeline-sort-dropdown">
            <button class="flex items-center space-x-2 text-sm font-medium text-gray-700 hover:text-blue-600 focus:outline-none">
              <span>Sort by: {{ sortLabel }}</span>
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
            </button>
            <div class="absolute right-0 top-full mt-2 w-48 bg-white rounded-md shadow-lg border border-gray-100 py-1 hidden group-hover:block z-10">
              <div id="pipeline-sort-option-recent" @click="sortBy = 'recent'" class="px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 cursor-pointer">Most Recent</div>
              <div id="pipeline-sort-option-status" @click="sortBy = 'status'" class="px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 cursor-pointer">Status</div>
            </div>
          </div>
        </div>

        <!-- Pipeline List -->
        <div id="pipeline-list-container" class="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden min-h-[500px]">
          <div id="pipeline-list" class="divide-y divide-gray-100">
            <div 
              v-for="pipe in filteredPipelines" 
              :key="pipe.id"
              class="group p-4 flex items-center space-x-4 hover:bg-blue-50 transition-colors cursor-pointer"
              :class="{
                'pipeline-row-filtered': hasFilters,
                'pipeline-row-matched': hasSearched && matchesSearch(pipe),
                'pipeline-row-visible': !hasFilters && !hasSearched
              }"
              @click="openPipeline(pipe)"
            >
              <!-- Status Icon -->
              <div class="flex-shrink-0 w-8 h-8 flex items-center justify-center rounded-full"
                :class="{
                  'bg-green-100 text-green-600': pipe.status === 'success',
                  'bg-red-100 text-red-600': pipe.status === 'failed',
                  'bg-blue-100 text-blue-600 animate-pulse': pipe.status === 'running'
                }"
              >
                <svg v-if="pipe.status === 'success'" class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>
                <svg v-else-if="pipe.status === 'failed'" class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
                <svg v-else class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/></svg>
              </div>

              <!-- Pipeline Info -->
              <div class="flex-1 min-w-0">
                <div class="flex items-center justify-between mb-1">
                  <h3 class="text-base font-semibold text-gray-900 truncate group-hover:text-blue-600" :class="`data-id-${pipe.id}`">
                    {{ pipe.name }} <span class="text-gray-400 font-normal">#{{ pipe.id.split('_')[1] }}</span>
                  </h3>
                  <span class="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">{{ pipe.branch }}</span>
                </div>
                <div class="flex items-center text-sm text-gray-600">
                  <div class="flex items-center mr-4">
                     <svg class="w-4 h-4 mr-1 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
                     {{ pipe.trigger }}
                  </div>
                  <div class="flex items-center">
                     <svg class="w-4 h-4 mr-1 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
                     {{ pipe.created_at }}
                  </div>
                </div>
              </div>
              
              <!-- Image (Build Artifact preview / Screen) -->
              <div class="hidden sm:block w-16 h-12 rounded bg-gray-100 overflow-hidden border border-gray-200">
                 <img :src="pipe.image" alt="artifact" class="w-full h-full object-cover opacity-80" />
              </div>
            </div>
            
            <!-- Empty State -->
            <div v-if="filteredPipelines.length === 0" class="p-12 text-center text-gray-500">
               <img src="/images/NoPipelines.jpg" alt="No pipelines found" class="w-32 h-32 mx-auto mb-4 opacity-50">
               <p class="text-lg font-medium">No pipelines found</p>
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
  name: 'PIPELINE_LIST',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const searchQuery = ref('')
    const filterFailed = ref(false)
    const sortBy = ref(null)

    // Date logic
    // created_at is string like '2023-10-03 10:00'. parse it.
    const dates = dataStore.pipelines.map(p => new Date(p.created_at).getTime())
    const minDate = dates.length ? Math.min(...dates) : 0
    const maxDate = dates.length ? Math.max(...dates) : 0
    const filterDate = ref(minDate)

    const sortLabel = computed(() => {
      if (sortBy.value === 'recent') return 'Most Recent'
      if (sortBy.value === 'status') return 'Status'
      return 'Default'
    })

    const filteredPipelines = computed(() => {
      let result = dataStore.pipelines

      if (searchQuery.value) {
        const q = searchQuery.value.toLowerCase()
        result = result.filter(p => p.name.toLowerCase().includes(q))
      }

      if (filterFailed.value) {
        result = result.filter(p => p.status === 'failed')
      }

      if (filterDate.value > minDate) {
        result = result.filter(p => new Date(p.created_at).getTime() >= filterDate.value)
      }

      if (sortBy.value === 'recent') {
        result = [...result].sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
      } else if (sortBy.value === 'status') {
        result = [...result].sort((a, b) => a.status.localeCompare(b.status))
      }

      return result
    })

    const hasFilters = computed(() => filterFailed.value || filterDate.value > minDate || sortBy.value !== null)
    const hasSearched = computed(() => searchQuery.value.length > 0)
    
    const matchesSearch = (pipe) => {
      if (!searchQuery.value) return false
      return pipe.name.toLowerCase().includes(searchQuery.value.toLowerCase())
    }

    const handleSearch = () => {
      signatureStore.pipeline_list_has_searched = true
      signatureStore.matched_pipeline_id = filteredPipelines.value.length > 0 ? filteredPipelines.value[0].id : null
    }

    const openPipeline = async (pipe) => {
      signatureStore.selected_pipeline_id = pipe.id
      
      if (hasFilters.value) {
        signatureStore.pipeline_list_filters_applied = true
      }
      if (hasSearched.value) {
        signatureStore.pipeline_list_has_searched = true
        signatureStore.matched_pipeline_id = pipe.id
      }
      if (!hasFilters.value && !hasSearched.value) {
        signatureStore.pipeline_list_viewport_anchor_id = pipe.id
      }

      await router.push({ name: 'PIPELINE_DETAIL', params: { pipeline_id: pipe.id } })
    }

    const goToConfigure = async () => {
      signatureStore.currentPageId = 'PIPELINE_CONFIG_FORM'
      await router.push({ name: 'PIPELINE_CONFIG_FORM' })
    }

    const goHome = async () => {
      signatureStore.currentPageId = 'HOME'
      await router.push({ name: 'HOME' })
    }

    return {
      searchQuery,
      filterFailed,
      filterDate,
      minDate,
      maxDate,
      sortBy,
      sortLabel,
      filteredPipelines,
      hasFilters,
      hasSearched,
      matchesSearch,
      handleSearch,
      openPipeline,
      goToConfigure,
      goHome
    }
  }
}
</script>