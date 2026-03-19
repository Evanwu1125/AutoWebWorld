<template>
  <div class="min-h-screen bg-[#FAFBFC] flex flex-col">
    <!-- Header -->
    <header class="bg-white border-b border-gray-200 px-6 py-4 flex justify-between items-center sticky top-0 z-20">
      <div class="flex items-center space-x-4">
        <button id="global-back-home" @click="goHome" class="text-gray-500 hover:text-blue-600 transition-colors">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"/></svg>
        </button>
        <h1 class="text-2xl font-bold text-[#172B4D]">Repositories</h1>
      </div>
      <button 
        id="create-repo-button" 
        @click="goToCreateRepo"
        class="bg-[#0052CC] text-white px-4 py-2 rounded-md font-medium hover:bg-blue-700 transition-colors shadow-sm flex items-center"
      >
        <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"/></svg>
        Create repository
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
                id="repo-search-input"
                v-model="searchQuery"
                @keyup.enter="handleSearch"
                type="text" 
                placeholder="Find a repository..."
                class="w-full pl-9 pr-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
              <svg class="w-4 h-4 text-gray-400 absolute left-3 top-2.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
            </div>
          </div>

          <!-- Checkboxes -->
          <div class="space-y-3 mb-6">
            <label class="flex items-center space-x-2 cursor-pointer text-sm text-gray-700 hover:text-blue-600">
              <input type="checkbox" id="filter-own-repos-checkbox" v-model="filterOwn" class="form-checkbox text-blue-600 rounded">
              <span>My repositories</span>
            </label>
            <label class="flex items-center space-x-2 cursor-pointer text-sm text-gray-700 hover:text-blue-600">
              <input type="checkbox" id="filter-private-repos-checkbox" v-model="filterPrivate" class="form-checkbox text-blue-600 rounded">
              <span>Private only</span>
            </label>
          </div>

          <!-- Slider -->
          <div class="mb-6">
            <label class="block text-sm font-medium text-gray-700 mb-2">
              Min Activity Score: {{ filterActivity }}
            </label>
            <input 
              id="activity-slider"
              type="range" 
              v-model.number="filterActivity" 
              :min="minActivity" 
              :max="maxActivity"
              step="1"
              class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
            >
            <div class="flex justify-between text-xs text-gray-500 mt-1">
              <span>{{ minActivity }}</span>
              <span>{{ maxActivity }}</span>
            </div>
          </div>
        </div>
      </aside>

      <!-- Main Content -->
      <div class="flex-1">
        <!-- Toolbar -->
        <div class="flex justify-between items-center mb-6">
          <div class="text-sm text-gray-500">
            Showing <span class="font-bold text-gray-900">{{ filteredRepos.length }}</span> repositories
          </div>
          
          <!-- Sort Dropdown -->
          <div class="relative group" id="repo-sort-dropdown">
            <button class="flex items-center space-x-2 text-sm font-medium text-gray-700 hover:text-blue-600 focus:outline-none">
              <span>Sort by: {{ sortLabel }}</span>
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
            </button>
            <div class="absolute right-0 top-full mt-2 w-48 bg-white rounded-md shadow-lg border border-gray-100 py-1 hidden group-hover:block z-10">
              <div id="sort-option-recent" @click="sortBy = 'recent'" class="px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 cursor-pointer">Most Recent</div>
              <div id="sort-option-name" @click="sortBy = 'name'" class="px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 cursor-pointer">Name (A-Z)</div>
              <div id="sort-option-owner" @click="sortBy = 'owner'" class="px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 cursor-pointer">Owner</div>
            </div>
          </div>
        </div>

        <!-- Repo List -->
        <div id="repo-list-container" class="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden min-h-[500px]">
          <div id="repo-list" class="divide-y divide-gray-100">
            <div 
              v-for="repo in filteredRepos" 
              :key="repo.id"
              class="group p-4 flex items-start space-x-4 hover:bg-blue-50 transition-colors cursor-pointer"
              :class="{
                'repo-row-filtered': hasFilters,
                'repo-row-matched': hasSearched && matchesSearch(repo),
                'repo-row-visible': !hasFilters && !hasSearched
              }"
              @click="openRepo(repo)"
            >
              <!-- Repo Icon/Image -->
              <div class="flex-shrink-0 w-12 h-12 rounded-lg overflow-hidden bg-gray-100 border border-gray-200">
                <img :src="repo.image" :alt="repo.name" class="w-full h-full object-cover">
              </div>
              
              <div class="flex-1 min-w-0">
                <div class="flex items-center justify-between mb-1">
                  <h3 class="text-base font-semibold text-blue-600 truncate group-hover:underline" :class="`data-id-${repo.id}`">
                    {{ repo.owner }} / {{ repo.name }}
                  </h3>
                  <span 
                    class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium"
                    :class="repo.access === 'private' ? 'bg-gray-100 text-gray-800' : 'bg-green-100 text-green-800'"
                  >
                    <svg v-if="repo.access === 'private'" class="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"/></svg>
                    <svg v-else class="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
                    {{ repo.access }}
                  </span>
                </div>
                <p class="text-sm text-gray-600 mb-2 line-clamp-2">{{ repo.description }}</p>
                <div class="flex items-center text-xs text-gray-500 space-x-4">
                  <span>Updated {{ repo.updated_at }}</span>
                  <span class="flex items-center">
                    <svg class="w-3 h-3 mr-1 text-yellow-500" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>
                    Activity: {{ repo.activity }}
                  </span>
                </div>
              </div>
            </div>
            
            <!-- Empty State -->
            <div v-if="filteredRepos.length === 0" class="p-12 text-center text-gray-500">
               <img src="/images/NoRepositories.jpg" alt="No repositories found" class="w-32 h-32 mx-auto mb-4 opacity-50">
               <p class="text-lg font-medium">No repositories found</p>
               <p class="text-sm">Try adjusting your filters or search query.</p>
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
  name: 'REPO_LIST',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    // State
    const searchQuery = ref('')
    const filterOwn = ref(false)
    const filterPrivate = ref(false)
    const filterActivity = ref(0)
    const sortBy = ref(null) // 'recent', 'name', 'owner'

    // Derived State from Mock Data
    const minActivity = computed(() => Math.min(...dataStore.repositories.map(r => r.activity), 0))
    const maxActivity = computed(() => Math.max(...dataStore.repositories.map(r => r.activity), 100))
    
    // Sort Label
    const sortLabel = computed(() => {
      if (sortBy.value === 'recent') return 'Most Recent'
      if (sortBy.value === 'name') return 'Name'
      if (sortBy.value === 'owner') return 'Owner'
      return 'Default'
    })

    // Computed Filtered List
    const filteredRepos = computed(() => {
      let result = dataStore.repositories

      // Filter: Search
      if (searchQuery.value) {
        const q = searchQuery.value.toLowerCase()
        result = result.filter(r => 
          r.name.toLowerCase().includes(q) || 
          r.description.toLowerCase().includes(q) ||
          r.owner.toLowerCase().includes(q)
        )
      }

      // Filter: Own (Mocking user_001 as 'My')
      // Note: In mock data, owner is a string name, not ID. I'll map 'Team Alpha' to user_001 logic or just filter by owner name if available.
      // But FSM doesn't specify logged in user mapping deeply. 
      // I'll assume 'Team Alpha' is the "Own" team for demo purposes if not specified.
      // Actually data.js has 'owner' string. Let's filter by exact string match or just some logic.
      if (filterOwn.value) {
        // Simple logic: keep 50% of repos to simulate "My Repos"
        // Better: Filter by owner 'Team Alpha'
        result = result.filter(r => r.owner === 'Team Alpha')
      }

      // Filter: Private
      if (filterPrivate.value) {
        result = result.filter(r => r.access === 'private')
      }

      // Filter: Activity Slider
      if (filterActivity.value > 0) {
        result = result.filter(r => r.activity >= filterActivity.value)
      }

      // Sort
      if (sortBy.value === 'recent') {
        result = [...result].sort((a, b) => new Date(b.updated_at) - new Date(a.updated_at))
      } else if (sortBy.value === 'name') {
        result = [...result].sort((a, b) => a.name.localeCompare(b.name))
      } else if (sortBy.value === 'owner') {
        result = [...result].sort((a, b) => a.owner.localeCompare(b.owner))
      }

      return result
    })

    const hasFilters = computed(() => {
      return filterOwn.value || filterPrivate.value || filterActivity.value > 0 || sortBy.value !== null
    })

    const hasSearched = computed(() => {
      return searchQuery.value.length > 0
    })

    const matchesSearch = (repo) => {
      if (!searchQuery.value) return false
      const q = searchQuery.value.toLowerCase()
      return repo.name.toLowerCase().includes(q)
    }

    // Actions
    const handleSearch = () => {
      // Triggered by Enter key
      signatureStore.repo_list_has_searched = true
      signatureStore.matched_repo_id = filteredRepos.value.length > 0 ? filteredRepos.value[0].id : null
    }

    const openRepo = async (repo) => {
      signatureStore.selected_repo_id = repo.id
      
      // Update store based on interaction type (filter vs search vs plain)
      // This is to satisfy FSM effects for specific actions
      if (hasFilters.value) {
        signatureStore.repo_list_filters_applied = true
      }
      if (hasSearched.value) {
        signatureStore.repo_list_has_searched = true
        signatureStore.matched_repo_id = repo.id
      }
      // If neither, it's ACT_REPO_LIST_OPEN_ANY_REPO (viewport)
      if (!hasFilters.value && !hasSearched.value) {
        signatureStore.repo_list_viewport_anchor_id = repo.id
      }

      await router.push({ name: 'REPO_DETAIL', params: { repo_id: repo.id } })
    }

    const goToCreateRepo = async () => {
      signatureStore.currentPageId = 'CREATE_REPO_FORM'
      await router.push({ name: 'CREATE_REPO_FORM' })
    }

    const goHome = async () => {
      signatureStore.currentPageId = 'HOME'
      await router.push({ name: 'HOME' })
    }

    return {
      searchQuery,
      filterOwn,
      filterPrivate,
      filterActivity,
      minActivity,
      maxActivity,
      sortBy,
      sortLabel,
      filteredRepos,
      hasFilters,
      hasSearched,
      matchesSearch,
      handleSearch,
      openRepo,
      goToCreateRepo,
      goHome
    }
  }
}
</script>