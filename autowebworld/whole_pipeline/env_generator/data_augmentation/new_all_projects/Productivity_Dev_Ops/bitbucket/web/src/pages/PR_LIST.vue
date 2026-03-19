<template>
  <div class="min-h-screen bg-[#FAFBFC] flex flex-col">
    <!-- Header -->
    <header class="bg-white border-b border-gray-200 px-6 py-4 flex justify-between items-center sticky top-0 z-20">
      <div class="flex items-center space-x-4">
        <button id="pr-list-back-home" @click="goHome" class="text-gray-500 hover:text-blue-600 transition-colors">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"/></svg>
        </button>
        <h1 class="text-2xl font-bold text-[#172B4D]">Pull Requests</h1>
      </div>
      <button 
        id="create-pr-button" 
        @click="goToCreatePR"
        class="bg-[#0052CC] text-white px-4 py-2 rounded-md font-medium hover:bg-blue-700 transition-colors shadow-sm flex items-center"
      >
        <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"/></svg>
        Create pull request
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
                id="pr-search-input"
                v-model="searchQuery"
                @keyup.enter="handleSearch"
                type="text" 
                placeholder="Find a pull request..."
                class="w-full pl-9 pr-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
              <svg class="w-4 h-4 text-gray-400 absolute left-3 top-2.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
            </div>
          </div>

          <!-- Checkboxes -->
          <div class="space-y-3 mb-6">
            <label class="flex items-center space-x-2 cursor-pointer text-sm text-gray-700 hover:text-blue-600">
              <input type="checkbox" id="filter-open-pr-checkbox" v-model="filterOpen" class="form-checkbox text-blue-600 rounded">
              <span>Open pull requests</span>
            </label>
            <label class="flex items-center space-x-2 cursor-pointer text-sm text-gray-700 hover:text-blue-600">
              <input type="checkbox" id="filter-my-pr-checkbox" v-model="filterMine" class="form-checkbox text-blue-600 rounded">
              <span>Created by me</span>
            </label>
          </div>

          <!-- Slider (Updated Date: Left means older, Right means newer?? FSM says drag left. Let's interpret slider value as "days ago" or "freshness")
               FSM Action: ACT_PR_LIST_FILTER_UPDATED_SLIDER (drag to left). 
               Interpretation: Filter logic for dates usually involves "Updated since X days ago".
               Slider value 0 = All (or oldest). Slider Max = Today.
               FSM drag left usually implies reducing value. 
               Let's map slider to "Days Since Update". 
               Or simply: Slider filters by `updated_at` timestamp.
               Let's use a numeric representation of date.
          -->
          <div class="mb-6">
            <label class="block text-sm font-medium text-gray-700 mb-2">
              Updated After: {{ new Date(filterDate).toLocaleDateString() }}
            </label>
            <input 
              id="pr-updated-slider"
              type="range" 
              v-model.number="filterDate" 
              :min="minDate" 
              :max="maxDate"
              step="86400000"
              class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
            >
             <!-- 86400000 = 1 day in ms -->
          </div>
        </div>
      </aside>

      <!-- Main Content -->
      <div class="flex-1">
        <!-- Toolbar -->
        <div class="flex justify-between items-center mb-6">
          <div class="text-sm text-gray-500">
            Showing <span class="font-bold text-gray-900">{{ filteredPRs.length }}</span> pull requests
          </div>
          
          <!-- Sort Dropdown -->
          <div class="relative group" id="pr-sort-dropdown">
            <button class="flex items-center space-x-2 text-sm font-medium text-gray-700 hover:text-blue-600 focus:outline-none">
              <span>Sort by: {{ sortLabel }}</span>
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
            </button>
            <div class="absolute right-0 top-full mt-2 w-48 bg-white rounded-md shadow-lg border border-gray-100 py-1 hidden group-hover:block z-10">
              <div id="pr-sort-option-recent-desc" @click="sortBy = 'recent'" class="px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 cursor-pointer">Most Recent</div>
              <div id="pr-sort-option-oldest-inc" @click="sortBy = 'oldest'" class="px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 cursor-pointer">Oldest First</div>
              <div id="pr-sort-option-title-inc" @click="sortBy = 'title'" class="px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 cursor-pointer">Title (A-Z)</div>
            </div>
          </div>
        </div>

        <!-- PR List -->
        <div id="pr-list-container" class="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden min-h-[500px]">
          <div id="pr-list" class="divide-y divide-gray-100">
            <div 
              v-for="pr in filteredPRs" 
              :key="pr.id"
              class="group p-4 flex items-start space-x-4 hover:bg-blue-50 transition-colors cursor-pointer"
              :class="{
                'pr-row-filtered': hasFilters,
                'pr-row-matched': hasSearched && matchesSearch(pr),
                'pr-row-visible': !hasFilters && !hasSearched
              }"
              @click="openPR(pr)"
            >
              <!-- Avatar -->
              <div class="flex-shrink-0 w-10 h-10 rounded-full overflow-hidden bg-gray-100 border border-gray-200">
                <img :src="pr.image" alt="pr avatar" class="w-full h-full object-cover">
              </div>
              
              <div class="flex-1 min-w-0">
                <div class="flex items-center justify-between mb-1">
                  <h3 class="text-base font-semibold text-gray-900 truncate group-hover:text-blue-600" :class="`data-id-${pr.id}`">
                    {{ pr.title }}
                  </h3>
                  <span 
                    class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium uppercase"
                    :class="{
                      'bg-green-100 text-green-800': pr.status === 'open',
                      'bg-purple-100 text-purple-800': pr.status === 'merged',
                      'bg-red-100 text-red-800': pr.status === 'declined'
                    }"
                  >
                    {{ pr.status }}
                  </span>
                </div>
                <p class="text-sm text-gray-600 mb-2">
                  <span class="font-medium text-gray-800">#{{ pr.id.split('_')[1] }}</span> created by {{ pr.author_id }} in {{ pr.repo_id }}
                </p>
                <div class="flex items-center text-xs text-gray-500">
                  <span>Updated {{ pr.updated_at }}</span>
                </div>
              </div>
            </div>
            
            <!-- Empty State -->
            <div v-if="filteredPRs.length === 0" class="p-12 text-center text-gray-500">
               <img src="/images/Nopullrequests.jpg" alt="No PRs found" class="w-32 h-32 mx-auto mb-4 opacity-50">
               <p class="text-lg font-medium">No pull requests found</p>
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
  name: 'PR_LIST',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    // State
    const searchQuery = ref('')
    const filterOpen = ref(false)
    const filterMine = ref(false)
    const sortBy = ref(null)

    // Date logic for slider
    // Find min and max dates in data
    const dates = dataStore.pull_requests.map(pr => new Date(pr.updated_at).getTime())
    const minDate = Math.min(...dates)
    const maxDate = Math.max(...dates)
    const filterDate = ref(minDate) // Default to showing everything (slider at start)

    // Sort Label
    const sortLabel = computed(() => {
      if (sortBy.value === 'recent') return 'Most Recent'
      if (sortBy.value === 'oldest') return 'Oldest First'
      if (sortBy.value === 'title') return 'Title'
      return 'Default'
    })

    const filteredPRs = computed(() => {
      let result = dataStore.pull_requests

      // Search
      if (searchQuery.value) {
        const q = searchQuery.value.toLowerCase()
        result = result.filter(pr => pr.title.toLowerCase().includes(q))
      }

      // Filter: Open
      if (filterOpen.value) {
        result = result.filter(pr => pr.status === 'open')
      }

      // Filter: Mine (user_001)
      if (filterMine.value) {
        result = result.filter(pr => pr.author_id === 'user_001')
      }

      // Filter: Date Slider (Updated After X)
      // Check if slider moved from min
      if (filterDate.value > minDate) {
         result = result.filter(pr => new Date(pr.updated_at).getTime() >= filterDate.value)
      }

      // Sort
      if (sortBy.value === 'recent') {
        result = [...result].sort((a, b) => new Date(b.updated_at) - new Date(a.updated_at))
      } else if (sortBy.value === 'oldest') {
        result = [...result].sort((a, b) => new Date(a.updated_at) - new Date(b.updated_at))
      } else if (sortBy.value === 'title') {
        result = [...result].sort((a, b) => a.title.localeCompare(b.title))
      }

      return result
    })

    const hasFilters = computed(() => {
      return filterOpen.value || filterMine.value || filterDate.value > minDate || sortBy.value !== null
    })

    const hasSearched = computed(() => {
      return searchQuery.value.length > 0
    })

    const matchesSearch = (pr) => {
      if (!searchQuery.value) return false
      return pr.title.toLowerCase().includes(searchQuery.value.toLowerCase())
    }

    // Actions
    const handleSearch = () => {
      signatureStore.pr_list_has_searched = true
      signatureStore.matched_pr_id = filteredPRs.value.length > 0 ? filteredPRs.value[0].id : null
    }

    const openPR = async (pr) => {
      signatureStore.selected_pr_id = pr.id
      
      if (hasFilters.value) {
        signatureStore.pr_list_filters_applied = true
      }
      if (hasSearched.value) {
        signatureStore.pr_list_has_searched = true
        signatureStore.matched_pr_id = pr.id
      }
      if (!hasFilters.value && !hasSearched.value) {
        signatureStore.pr_list_viewport_anchor_id = pr.id
      }

      await router.push({ name: 'PR_DETAIL', params: { pr_id: pr.id } })
    }

    const goToCreatePR = async () => {
      signatureStore.currentPageId = 'CREATE_PR_FORM'
      await router.push({ name: 'CREATE_PR_FORM' })
    }

    const goHome = async () => {
      signatureStore.currentPageId = 'HOME'
      await router.push({ name: 'HOME' })
    }

    return {
      searchQuery,
      filterOpen,
      filterMine,
      filterDate,
      minDate,
      maxDate,
      sortBy,
      sortLabel,
      filteredPRs,
      hasFilters,
      hasSearched,
      matchesSearch,
      handleSearch,
      openPR,
      goToCreatePR,
      goHome
    }
  }
}
</script>