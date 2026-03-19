<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <!-- Top Bar -->
    <header class="bg-white border-b border-gray-200 px-6 py-3 flex items-center justify-between sticky top-0 z-30 shadow-sm">
       <div class="flex items-center gap-4">
         <button id="back-home" @click="goHome" class="p-2 text-gray-500 hover:text-blue-600 rounded-full hover:bg-blue-50 transition-colors">
           <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
             <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
           </svg>
         </button>
         <h1 class="text-xl font-bold text-gray-900">Workspaces</h1>
       </div>
       
       <div class="flex items-center gap-4">
          <div class="relative">
            <input 
              id="bases-search-input"
              v-model="searchQuery"
              @keyup.enter="handleSearch"
              type="text" 
              placeholder="Find a base..." 
              class="pl-10 pr-4 py-2 border border-gray-300 rounded-full text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent w-64 transition-all"
            >
            <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-gray-400 absolute left-3 top-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>
          
          <button id="create-base-button" @click="goToCreateBase" class="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-md shadow-sm transition-all flex items-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
               <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
            </svg>
            Create Base
          </button>
          
          <div class="w-8 h-8 rounded-full bg-indigo-100 flex items-center justify-center text-indigo-700 font-bold border border-indigo-200">
             U
          </div>
       </div>
    </header>

    <div class="flex flex-1 overflow-hidden">
      <!-- Sidebar Filters -->
      <aside class="w-72 bg-white border-r border-gray-200 p-6 overflow-y-auto hidden md:block">
         <h3 class="text-xs font-bold text-gray-400 uppercase tracking-wider mb-6">Filters & Sorting</h3>
         
         <div class="space-y-8">
           <!-- Starred Filter -->
           <div>
             <label class="flex items-center gap-3 cursor-pointer group">
               <div id="filter-starred-checkbox" 
                    class="w-5 h-5 border-2 border-gray-300 rounded transition-colors flex items-center justify-center group-hover:border-blue-400"
                    :class="{'bg-blue-600 border-blue-600': filterStarred}"
                    @click="toggleStarredFilter">
                 <svg v-if="filterStarred" xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5 text-white" viewBox="0 0 20 20" fill="currentColor">
                   <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
                 </svg>
               </div>
               <span class="text-sm font-medium text-gray-700 group-hover:text-gray-900">Starred Only</span>
             </label>
           </div>

           <!-- Activity Slider -->
           <div>
             <div class="flex justify-between mb-2">
               <label class="text-sm font-medium text-gray-700">Min Activity Level</label>
               <span class="text-xs font-semibold text-blue-600 bg-blue-50 px-2 py-0.5 rounded">{{ filterActivity }}%</span>
             </div>
             <input 
               id="activity-slider"
               type="range" 
               min="0" 
               max="100" 
               step="5"
               v-model.number="filterActivity"
               @input="handleActivityChange"
               class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
             >
             <div class="flex justify-between mt-1">
               <span class="text-xs text-gray-400">0%</span>
               <span class="text-xs text-gray-400">100%</span>
             </div>
           </div>
           
           <!-- Sort Dropdown -->
           <div class="relative">
             <label class="text-sm font-medium text-gray-700 block mb-2">Sort By</label>
             <button 
               id="bases-sort-dropdown"
               @click="sortOpen = !sortOpen"
               class="w-full flex items-center justify-between px-3 py-2 bg-white border border-gray-300 rounded-md text-sm text-gray-700 hover:border-blue-400 transition-colors shadow-sm"
             >
               <span>{{ sortLabel }}</span>
               <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                 <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
               </svg>
             </button>
             
             <div v-if="sortOpen" class="absolute top-full left-0 w-full mt-1 bg-white border border-gray-200 rounded-md shadow-xl z-20 overflow-hidden">
               <div id="bases-sort-recent-desc" @click="setSort('recent')" class="px-4 py-2 hover:bg-blue-50 text-sm cursor-pointer text-gray-700 flex justify-between items-center group">
                 <span>Recently Viewed</span>
                 <svg v-if="sortBy === 'recent'" xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-blue-600" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" /></svg>
               </div>
               <div id="bases-sort-alpha" @click="setSort('alphabetical')" class="px-4 py-2 hover:bg-blue-50 text-sm cursor-pointer text-gray-700 flex justify-between items-center group">
                 <span>Alphabetical (A-Z)</span>
                 <svg v-if="sortBy === 'alphabetical'" xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-blue-600" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" /></svg>
               </div>
               <div id="bases-sort-starred" @click="setSort('starred')" class="px-4 py-2 hover:bg-blue-50 text-sm cursor-pointer text-gray-700 flex justify-between items-center group">
                 <span>Starred First</span>
                 <svg v-if="sortBy === 'starred'" xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-blue-600" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" /></svg>
               </div>
             </div>
           </div>
         </div>
      </aside>

      <!-- Main Grid -->
      <main id="bases-grid" class="flex-1 p-8 overflow-y-auto" @scroll="handleScroll">
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          
          <!-- Base Card Template -->
          <div 
            v-for="base in displayBases" 
            :key="base.id"
            :class="[
              'group bg-white rounded-xl shadow-sm border border-gray-200 hover:shadow-md transition-all cursor-pointer overflow-hidden flex flex-col',
              'data-id-' + base.id,
              cardClass(base)
            ]"
            @click="openBase(base)"
          >
             <!-- Card Cover Image -->
             <div class="h-32 bg-gray-100 relative overflow-hidden">
               <img :src="base.image" class="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105" alt="Base cover" />
               <div class="absolute inset-0 bg-black/10 group-hover:bg-black/0 transition-colors"></div>
               
               <!-- Icon Badge -->
               <div :class="`absolute -bottom-6 left-4 w-12 h-12 rounded-lg bg-${base.color}-500 flex items-center justify-center text-white shadow-lg border-2 border-white`">
                 <!-- Simple icons mapping -->
                 <span v-if="base.icon === 'grid'" class="text-xl">⊞</span>
                 <span v-else-if="base.icon === 'calendar'" class="text-xl">📅</span>
                 <span v-else-if="base.icon === 'users'" class="text-xl">👥</span>
                 <span v-else class="text-xl">📄</span>
               </div>
             </div>

             <!-- Card Content -->
             <div class="pt-8 pb-4 px-4 flex-1 flex flex-col">
               <div class="flex justify-between items-start mb-2">
                 <h3 class="font-bold text-gray-900 text-lg group-hover:text-blue-600 transition-colors line-clamp-1">{{ base.name }}</h3>
                 <span v-if="base.starred" class="text-yellow-400">★</span>
               </div>
               <p class="text-xs text-gray-500 mt-auto">Opened {{ formatDate(base.last_viewed) }}</p>
             </div>
          </div>

          <!-- Empty State -->
          <div v-if="displayBases.length === 0" class="col-span-full flex flex-col items-center justify-center py-20 text-center">
            <div class="w-24 h-24 bg-gray-100 rounded-full flex items-center justify-center mb-6">
              <svg xmlns="http://www.w3.org/2000/svg" class="h-10 w-10 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <h3 class="text-xl font-bold text-gray-900 mb-2">No bases found</h3>
            <p class="text-gray-500 max-w-sm">
              We couldn't find any bases matching your current filters. Try adjusting your search or filters.
            </p>
          </div>
        </div>
      </main>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'BASES_DASHBOARD',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    // Filters & Search State
    const filterStarred = ref(false)
    const filterActivity = ref(0)
    const sortBy = ref('')
    const sortOpen = ref(false)
    const searchQuery = ref('')

    // Helpers
    const formatDate = (dateString) => {
      const options = { month: 'short', day: 'numeric' }
      return new Date(dateString).toLocaleDateString(undefined, options)
    }

    const sortLabel = computed(() => {
      if (sortBy.value === 'recent') return 'Recently Viewed'
      if (sortBy.value === 'alphabetical') return 'Alphabetical'
      if (sortBy.value === 'starred') return 'Starred First'
      return 'Default'
    })

    // Filter Logic
    const displayBases = computed(() => {
      let result = [...store.bases]

      // 1. Search (ACT_BASES_SEARCH_BASE)
      if (store.bases_dashboard_has_searched && store.matched_base_id) {
        return result.filter(b => b.id === store.matched_base_id)
      }
      
      // Also support realtime search via input model
      if (searchQuery.value) {
        const q = searchQuery.value.toLowerCase()
        result = result.filter(b => b.name.toLowerCase().includes(q))
      }

      // 2. Filters (ACT_BASES_FILTER_*)
      if (filterStarred.value) {
        result = result.filter(b => b.starred)
      }
      
      if (filterActivity.value > 0) {
        result = result.filter(b => b.activity >= filterActivity.value)
      }

      // 3. Sorting (ACT_BASES_SORT)
      if (sortBy.value) {
        if (sortBy.value === 'alphabetical') {
          result.sort((a, b) => a.name.localeCompare(b.name))
        } else if (sortBy.value === 'starred') {
          result.sort((a, b) => (b.starred === a.starred) ? 0 : b.starred ? 1 : -1)
        } else if (sortBy.value === 'recent') {
          result.sort((a, b) => new Date(b.last_viewed) - new Date(a.last_viewed))
        }
      }

      return result
    })

    // Dynamic class for FSM selectors
    const cardClass = (base) => {
      if (store.bases_dashboard_has_searched && base.id === store.matched_base_id) {
        return 'base-card-matched'
      }
      if (store.bases_dashboard_filters_applied) {
        return 'base-card-filtered'
      }
      return 'base-card-visible'
    }

    // Actions
    const goHome = async () => {
      store.setCurrentPageId('HOME')
      await router.push({ name: 'HOME' })
    }

    const goToCreateBase = async () => {
      store.setCurrentPageId('BASE_CREATE')
      await router.push({ name: 'BASE_CREATE' })
    }

    const toggleStarredFilter = () => {
      // ACT_BASES_FILTER_BY_STARRED_CHECKBOX
      filterStarred.value = !filterStarred.value
      store.bases_dashboard_filters_applied = true
    }

    const handleActivityChange = () => {
      // ACT_BASES_FILTER_BY_ACTIVITY_SLIDER
      store.bases_dashboard_filters_applied = true
    }

    const setSort = (type) => {
      // ACT_BASES_SORT
      sortBy.value = type
      sortOpen.value = false
      store.bases_dashboard_filters_applied = true
    }

    const handleSearch = () => {
      // ACT_BASES_SEARCH_BASE
      const match = store.bases.find(b => b.name.toLowerCase().includes(searchQuery.value.toLowerCase()))
      if (match) {
        store.matched_base_id = match.id
        store.bases_dashboard_has_searched = true
      }
    }

    const openBase = async (base) => {
      // ACT_BASES_OPEN_*
      store.selected_base_id = base.id
      // Reset filter states as per FSM effects if needed (handled in FSM effects usually, but here manually)
      store.bases_dashboard_filters_applied = false
      store.bases_dashboard_has_searched = false
      
      store.setCurrentPageId('BASE_WORKSPACE')
      await router.push({ name: 'BASE_WORKSPACE' })
    }
    
    const handleScroll = () => {
       // ACT_BASES_SCROLL_BASE_INTO_VIEW - simplified placeholder
    }

    return {
      searchQuery,
      filterStarred,
      filterActivity,
      sortBy,
      sortOpen,
      sortLabel,
      displayBases,
      cardClass,
      
      goHome,
      goToCreateBase,
      toggleStarredFilter,
      handleActivityChange,
      setSort,
      handleSearch,
      openBase,
      handleScroll,
      formatDate
    }
  }
}
</script>