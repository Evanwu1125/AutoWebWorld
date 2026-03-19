<template>
  <div class="min-h-screen bg-slate-50">
    <!-- Header -->
    <header class="bg-white border-b border-slate-200">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <h1 class="text-2xl font-bold text-slate-900">Flows</h1>
        <button 
          id="breadcrumb-dashboard" 
          @click="goBackDashboard"
          class="text-sm font-medium text-slate-500 hover:text-blue-600 flex items-center"
        >
          <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
          Back to Dashboard
        </button>
      </div>
    </header>

    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      
      <!-- Toolbar -->
      <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8">
        <div class="flex-1 w-full md:w-auto relative">
          <input 
            id="flows-search-input"
            v-model="searchQuery"
            @keypress.enter="handleSearch"
            type="text" 
            placeholder="Search flows..." 
            class="w-full md:w-96 pl-10 pr-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
          <svg class="w-5 h-5 text-slate-400 absolute left-3 top-2.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
        </div>
        
        <button 
          id="btn-create-flow"
          @click="createFlow"
          class="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-colors shadow-sm"
        >
          Create Flow
        </button>
      </div>

      <!-- Filters Row -->
      <div class="bg-white rounded-lg shadow-sm p-4 mb-6 flex flex-wrap gap-6 items-center border border-slate-100">
        <span class="text-sm font-medium text-slate-500 uppercase tracking-wider">Filters:</span>
        
        <!-- Status Filter (Checkbox) -->
        <label class="flex items-center space-x-2 cursor-pointer">
          <input 
            id="filter-status-live-checkbox"
            type="checkbox" 
            v-model="filterLive"
            @change="handleFilterChange"
            class="form-checkbox h-4 w-4 text-blue-600 rounded border-slate-300 focus:ring-blue-500"
          />
          <span class="text-sm text-slate-700">Live Only</span>
        </label>

        <!-- Sort Dropdown -->
        <div class="relative ml-auto">
          <button 
            id="flows-sort-dropdown"
            @click="toggleSortDropdown"
            class="flex items-center space-x-1 text-sm font-medium text-slate-600 hover:text-slate-900"
          >
            <span>Sort by: {{ currentSortLabel }}</span>
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
          </button>
          
          <div v-if="sortDropdownOpen" class="absolute right-0 mt-2 w-40 bg-white rounded-lg shadow-xl z-50 border border-slate-100 py-1">
            <div 
              id="flows-sort-option-newest"
              @click="handleSort('newest')"
              class="block px-4 py-2 text-sm text-slate-700 hover:bg-slate-50 cursor-pointer"
            >
              Newest First
            </div>
            <div 
              id="flows-sort-option-oldest"
              @click="handleSort('oldest')"
              class="block px-4 py-2 text-sm text-slate-700 hover:bg-slate-50 cursor-pointer"
            >
              Oldest First
            </div>
          </div>
        </div>
      </div>

      <!-- Flows List -->
      <div id="flows-table" class="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
        <div class="grid grid-cols-12 gap-4 p-4 bg-slate-50 border-b border-slate-200 text-xs font-semibold text-slate-500 uppercase tracking-wider">
          <div class="col-span-6">Flow Name</div>
          <div class="col-span-2">Status</div>
          <div class="col-span-2">Trigger</div>
          <div class="col-span-2 text-right">Revenue</div>
        </div>

        <div v-if="filteredItems.length === 0" class="p-8 text-center text-slate-500">
          No flows found.
        </div>

        <div 
          v-for="item in filteredItems" 
          :key="item.id"
          class="grid grid-cols-12 gap-4 p-4 border-b border-slate-100 hover:bg-slate-50 transition-colors cursor-pointer group items-center"
          :class="{
            'row-filtered': isFiltered,
            'row-matched': isSearched && item.id === store.matched_flow_id,
            'row-visible': !isFiltered && !isSearched
          }"
          :data-id="item.id"
          @click="openItem(item)"
        >
          <div class="col-span-6 flex items-center space-x-4">
             <div class="w-10 h-10 rounded bg-emerald-100 flex items-center justify-center text-emerald-600 font-bold shrink-0">
               <img :src="item.image" :alt="item.name" class="w-full h-full object-cover rounded opacity-80" />
             </div>
            <span class="font-medium text-slate-900 group-hover:text-blue-600 transition-colors">{{ item.name }}</span>
          </div>
          <div class="col-span-2">
            <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium capitalize"
              :class="{
                'bg-emerald-100 text-emerald-800': item.status === 'live',
                'bg-slate-100 text-slate-800': item.status === 'draft'
              }">
              <span class="w-2 h-2 rounded-full mr-1.5" :class="item.status === 'live' ? 'bg-emerald-500' : 'bg-slate-400'"></span>
              {{ item.status }}
            </span>
          </div>
          <div class="col-span-2 text-sm text-slate-600">
            {{ item.trigger }}
          </div>
          <div class="col-span-2 text-right font-medium text-slate-900">
            ${{ item.revenue.toLocaleString() }}
          </div>
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
  name: 'FLOWS_LIST',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const searchQuery = ref('')
    const filterLive = ref(false)
    const sortDropdownOpen = ref(false)
    const currentSort = ref(null)

    const isFiltered = computed(() => store.flows_list_filters_applied === true)
    const isSearched = computed(() => store.flows_list_has_searched === true)

    const currentSortLabel = computed(() => {
      if (!currentSort.value) return 'Default'
      if (currentSort.value === 'newest') return 'Newest'
      if (currentSort.value === 'oldest') return 'Oldest'
      return 'Newest'
    })

    const filteredItems = computed(() => {
      let items = [...dataStore.flows]

      if (searchQuery.value) {
        const q = searchQuery.value.toLowerCase()
        items = items.filter(i => i.name.toLowerCase().includes(q))
      }

      if (filterLive.value) {
        items = items.filter(i => i.status === 'live')
      }

      // Sort (mock logic)
      if (currentSort.value === 'newest') {
        items.sort((a, b) => b.id.localeCompare(a.id)) // Mock newest by ID
      } else {
        items.sort((a, b) => a.id.localeCompare(b.id))
      }

      return items
    })

    function handleFilterChange() {
      store.flows_list_filters_applied = true
    }

    function handleSearch() {
      store.flows_list_has_searched = true
      const match = dataStore.flows.find(i => i.name.toLowerCase().includes(searchQuery.value.toLowerCase()))
      if (match) {
        store.matched_flow_id = match.id
      }
    }

    function toggleSortDropdown() {
      sortDropdownOpen.value = !sortDropdownOpen.value
    }

    function handleSort(type) {
      currentSort.value = type
      store.flows_list_filters_applied = true
      sortDropdownOpen.value = false
    }

    async function openItem(item) {
      store.selected_flow_id = item.id
      store.flows_list_has_searched = null
      store.flows_list_filters_applied = null
      store.setCurrentPageId('FLOW_DETAIL')
      await router.push({ name: 'FLOW_DETAIL', params: { id: item.id } })
    }

    async function createFlow() {
      store.setCurrentPageId('FLOW_TRIGGER_SETUP')
      await router.push({ name: 'FLOW_TRIGGER_SETUP' })
    }

    async function goBackDashboard() {
      store.setCurrentPageId('DASHBOARD')
      await router.push({ name: 'DASHBOARD' })
    }

    return {
      store,
      searchQuery,
      filterLive,
      sortDropdownOpen,
      currentSortLabel,
      filteredItems,
      isFiltered,
      isSearched,
      handleFilterChange,
      handleSearch,
      toggleSortDropdown,
      handleSort,
      openItem,
      createFlow,
      goBackDashboard
    }
  }
}
</script>