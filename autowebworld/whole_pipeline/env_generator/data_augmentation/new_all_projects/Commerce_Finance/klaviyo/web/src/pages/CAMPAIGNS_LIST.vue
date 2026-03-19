<template>
  <div class="min-h-screen bg-slate-50">
    <!-- Header -->
    <header class="bg-white border-b border-slate-200">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <h1 class="text-2xl font-bold text-slate-900">Campaigns</h1>
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
            id="campaigns-search-input"
            v-model="searchQuery"
            @keypress.enter="handleSearch"
            type="text" 
            placeholder="Search campaigns..." 
            class="w-full md:w-96 pl-10 pr-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
          <svg class="w-5 h-5 text-slate-400 absolute left-3 top-2.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
        </div>
        
        <button 
          id="btn-create-campaign"
          @click="createCampaign"
          class="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-colors shadow-sm"
        >
          Create Campaign
        </button>
      </div>

      <!-- Filters Row -->
      <div class="bg-white rounded-lg shadow-sm p-4 mb-6 flex flex-wrap gap-6 items-center border border-slate-100">
        <span class="text-sm font-medium text-slate-500 uppercase tracking-wider">Filters:</span>
        
        <!-- Status Filter (Checkbox) -->
        <label class="flex items-center space-x-2 cursor-pointer">
          <input 
            id="filter-status-scheduled-checkbox"
            type="checkbox" 
            v-model="filterScheduled"
            @change="handleFilterChange"
            class="form-checkbox h-4 w-4 text-blue-600 rounded border-slate-300 focus:ring-blue-500"
          />
          <span class="text-sm text-slate-700">Scheduled Only</span>
        </label>

        <!-- Revenue Slider -->
        <div class="flex items-center space-x-4">
          <span class="text-sm text-slate-700">Min Revenue: ${{ filterRevenue }}</span>
          <input 
            id="filter-revenue-slider"
            type="range" 
            v-model.number="filterRevenue"
            @input="handleFilterChange"
            :min="0"
            :max="maxRevenue"
            step="100"
            class="w-48 h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
          />
        </div>

        <!-- Sort Dropdown -->
        <div class="relative ml-auto">
          <button 
            id="campaigns-sort-dropdown"
            @click="toggleSortDropdown"
            class="flex items-center space-x-1 text-sm font-medium text-slate-600 hover:text-slate-900"
          >
            <span>Sort by: {{ currentSortLabel }}</span>
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
          </button>
          
          <div v-if="sortDropdownOpen" class="absolute right-0 mt-2 w-40 bg-white rounded-lg shadow-xl z-50 border border-slate-100 py-1">
            <div 
              id="campaigns-sort-option-newest"
              @click="handleSort('newest')"
              class="block px-4 py-2 text-sm text-slate-700 hover:bg-slate-50 cursor-pointer"
            >
              Newest First
            </div>
            <div 
              id="campaigns-sort-option-oldest"
              @click="handleSort('oldest')"
              class="block px-4 py-2 text-sm text-slate-700 hover:bg-slate-50 cursor-pointer"
            >
              Oldest First
            </div>
             <div 
              id="campaigns-sort-option-status"
              @click="handleSort('status')"
              class="block px-4 py-2 text-sm text-slate-700 hover:bg-slate-50 cursor-pointer"
            >
              Status
            </div>
          </div>
        </div>
      </div>

      <!-- Campaign List -->
      <div id="campaigns-table" class="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
        <div class="grid grid-cols-12 gap-4 p-4 bg-slate-50 border-b border-slate-200 text-xs font-semibold text-slate-500 uppercase tracking-wider">
          <div class="col-span-6">Campaign Info</div>
          <div class="col-span-2">Status</div>
          <div class="col-span-2">Date</div>
          <div class="col-span-2 text-right">Revenue</div>
        </div>

        <div v-if="filteredItems.length === 0" class="p-8 text-center text-slate-500">
          No campaigns found matching your criteria.
        </div>

        <div 
          v-for="item in filteredItems" 
          :key="item.id"
          class="grid grid-cols-12 gap-4 p-4 border-b border-slate-100 hover:bg-slate-50 transition-colors cursor-pointer group items-center"
          :class="{
            'row-filtered': isFiltered,
            'row-matched': isSearched && item.id === store.matched_campaign_id,
            'row-visible': !isFiltered && !isSearched
          }"
          :data-id="item.id"
          @click="openItem(item)"
        >
          <div class="col-span-6 flex items-center space-x-4">
            <div class="w-12 h-12 rounded-lg bg-slate-200 overflow-hidden flex-shrink-0">
              <img :src="item.image" :alt="item.name" class="w-full h-full object-cover" />
            </div>
            <div>
              <h3 class="font-semibold text-slate-900 group-hover:text-blue-600 transition-colors">{{ item.name }}</h3>
              <span class="text-xs px-2 py-0.5 rounded-full bg-slate-100 text-slate-600 border border-slate-200 uppercase tracking-wide">
                {{ item.type }}
              </span>
            </div>
          </div>
          <div class="col-span-2">
            <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium capitalize"
              :class="{
                'bg-green-100 text-green-800': item.status === 'sent',
                'bg-blue-100 text-blue-800': item.status === 'scheduled',
                'bg-slate-100 text-slate-800': item.status === 'draft'
              }">
              {{ item.status }}
            </span>
          </div>
          <div class="col-span-2 text-sm text-slate-600">
            {{ item.sent || 'Not sent' }}
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
  name: 'CAMPAIGNS_LIST',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const searchQuery = ref('')
    const filterScheduled = ref(false)
    const filterRevenue = ref(0)
    const sortDropdownOpen = ref(false)
    const currentSort = ref(null)

    // Calculated Max Revenue for Slider
    const maxRevenue = computed(() => {
      return Math.max(...dataStore.campaigns.map(c => c.revenue), 10000)
    })

    const isFiltered = computed(() => store.campaigns_list_filters_applied === true)
    const isSearched = computed(() => store.campaigns_list_has_searched === true)

    const currentSortLabel = computed(() => {
      if (!currentSort.value) return 'Default'
      if (currentSort.value === 'newest') return 'Newest'
      if (currentSort.value === 'oldest') return 'Oldest'
      if (currentSort.value === 'status') return 'Status'
      return 'Newest'
    })

    const filteredItems = computed(() => {
      let items = [...dataStore.campaigns]

      // Search
      if (searchQuery.value) {
        const q = searchQuery.value.toLowerCase()
        items = items.filter(i => i.name.toLowerCase().includes(q))
      }

      // Filter: Scheduled
      if (filterScheduled.value) {
        items = items.filter(i => i.status === 'scheduled')
      }

      // Filter: Revenue (Greater than)
      items = items.filter(i => i.revenue >= filterRevenue.value)

      // Sort
      if (currentSort.value === 'newest') {
        // Mock date sorting (sent date, nulls last)
        items.sort((a, b) => (b.sent || '').localeCompare(a.sent || ''))
      } else if (currentSort.value === 'oldest') {
        items.sort((a, b) => (a.sent || '').localeCompare(b.sent || ''))
      } else if (currentSort.value === 'status') {
        items.sort((a, b) => a.status.localeCompare(b.status))
      }

      return items
    })

    function handleFilterChange() {
      store.campaigns_list_filters_applied = true
    }

    function handleSearch() {
      store.campaigns_list_has_searched = true
      // Find exact match ID if possible for effect
      const match = dataStore.campaigns.find(i => i.name.toLowerCase().includes(searchQuery.value.toLowerCase()))
      if (match) {
        store.matched_campaign_id = match.id
      }
    }

    function toggleSortDropdown() {
      sortDropdownOpen.value = !sortDropdownOpen.value
    }

    function handleSort(type) {
      currentSort.value = type
      store.campaigns_list_filters_applied = true
      sortDropdownOpen.value = false
    }

    async function openItem(item) {
      store.selected_campaign_id = item.id
      store.matched_campaign_id = null // Clear matched state on navigation
      store.campaigns_list_has_searched = null
      store.campaigns_list_filters_applied = null
      store.setCurrentPageId('CAMPAIGN_DETAIL')
      await router.push({ name: 'CAMPAIGN_DETAIL', params: { id: item.id } })
    }

    async function createCampaign() {
      store.setCurrentPageId('CREATE_CAMPAIGN_CHANNEL')
      await router.push({ name: 'CREATE_CAMPAIGN_CHANNEL' })
    }

    async function goBackDashboard() {
      store.setCurrentPageId('DASHBOARD')
      await router.push({ name: 'DASHBOARD' })
    }

    return {
      store,
      searchQuery,
      filterScheduled,
      filterRevenue,
      maxRevenue,
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
      createCampaign,
      goBackDashboard
    }
  }
}
</script>