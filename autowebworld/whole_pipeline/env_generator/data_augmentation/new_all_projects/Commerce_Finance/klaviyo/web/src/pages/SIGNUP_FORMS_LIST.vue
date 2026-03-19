<template>
  <div class="min-h-screen bg-slate-50">
    <!-- Header -->
    <header class="bg-white border-b border-slate-200">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <h1 class="text-2xl font-bold text-slate-900">Signup Forms</h1>
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
            id="forms-search-input"
            v-model="searchQuery"
            @keypress.enter="handleSearch"
            type="text" 
            placeholder="Search forms..." 
            class="w-full md:w-96 pl-10 pr-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
          <svg class="w-5 h-5 text-slate-400 absolute left-3 top-2.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
        </div>
        
        <button 
          id="btn-create-form"
          @click="createForm"
          class="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-colors shadow-sm"
        >
          Create Signup Form
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
            id="forms-sort-dropdown"
            @click="toggleSortDropdown"
            class="flex items-center space-x-1 text-sm font-medium text-slate-600 hover:text-slate-900"
          >
            <span>Sort by: {{ currentSortLabel }}</span>
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
          </button>
          
          <div v-if="sortDropdownOpen" class="absolute right-0 mt-2 w-40 bg-white rounded-lg shadow-xl z-50 border border-slate-100 py-1">
            <div 
              id="forms-sort-option-newest"
              @click="handleSort('newest')"
              class="block px-4 py-2 text-sm text-slate-700 hover:bg-slate-50 cursor-pointer"
            >
              Newest First
            </div>
            <div 
              id="forms-sort-option-oldest"
              @click="handleSort('oldest')"
              class="block px-4 py-2 text-sm text-slate-700 hover:bg-slate-50 cursor-pointer"
            >
              Oldest First
            </div>
          </div>
        </div>
      </div>

      <!-- Forms Grid/List -->
      <div id="forms-table" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        
        <div v-if="filteredItems.length === 0" class="col-span-full p-8 text-center text-slate-500 bg-white rounded-lg border border-slate-200">
          No signup forms found.
        </div>

        <div 
          v-for="item in filteredItems" 
          :key="item.id"
          class="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden hover:shadow-md transition-shadow cursor-pointer group"
          :class="{
            'row-filtered': isFiltered,
            'row-matched': isSearched && item.id === store.matched_form_id,
            'row-visible': !isFiltered && !isSearched
          }"
          :data-id="item.id"
          @click="openItem(item)"
        >
          <div class="h-40 relative bg-slate-100">
            <img :src="item.image" :alt="item.name" class="w-full h-full object-cover opacity-90 group-hover:opacity-100 transition-opacity" />
            <div class="absolute top-2 right-2">
               <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-bold capitalize shadow-sm"
                :class="{
                  'bg-orange-100 text-orange-800': item.status === 'live',
                  'bg-slate-100 text-slate-800': item.status === 'draft' || item.status === 'paused'
                }">
                {{ item.status }}
              </span>
            </div>
          </div>
          <div class="p-4">
            <h3 class="text-lg font-bold text-slate-900 mb-1 group-hover:text-blue-600 transition-colors">{{ item.name }}</h3>
            <p class="text-sm text-slate-500 capitalize mb-4">{{ item.type.replace('_', ' ') }}</p>
            
            <div class="flex justify-between items-center text-xs text-slate-400 border-t border-slate-100 pt-3">
              <div>{{ item.views.toLocaleString() }} views</div>
              <div>{{ item.submissions.toLocaleString() }} submissions</div>
            </div>
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
  name: 'SIGNUP_FORMS_LIST',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const searchQuery = ref('')
    const filterLive = ref(false)
    const sortDropdownOpen = ref(false)
    const currentSort = ref(null)

    const isFiltered = computed(() => store.signup_forms_filters_applied === true)
    const isSearched = computed(() => store.signup_forms_has_searched === true)

    const currentSortLabel = computed(() => {
      if (!currentSort.value) return 'Default'
      if (currentSort.value === 'newest') return 'Newest'
      if (currentSort.value === 'oldest') return 'Oldest'
      return 'Newest'
    })

    const filteredItems = computed(() => {
      let items = [...dataStore.signup_forms]

      if (searchQuery.value) {
        const q = searchQuery.value.toLowerCase()
        items = items.filter(i => i.name.toLowerCase().includes(q))
      }

      if (filterLive.value) {
        items = items.filter(i => i.status === 'live')
      }

      // Sort
      if (currentSort.value === 'newest') {
        items.sort((a, b) => b.id.localeCompare(a.id))
      } else {
        items.sort((a, b) => a.id.localeCompare(b.id))
      }

      return items
    })

    function handleFilterChange() {
      store.signup_forms_filters_applied = true
    }

    function handleSearch() {
      store.signup_forms_has_searched = true
      const match = dataStore.signup_forms.find(i => i.name.toLowerCase().includes(searchQuery.value.toLowerCase()))
      if (match) {
        store.matched_form_id = match.id
      }
    }

    function toggleSortDropdown() {
      sortDropdownOpen.value = !sortDropdownOpen.value
    }

    function handleSort(type) {
      currentSort.value = type
      store.signup_forms_filters_applied = true
      sortDropdownOpen.value = false
    }

    async function openItem(item) {
      store.selected_form_id = item.id
      store.signup_forms_has_searched = null
      store.signup_forms_filters_applied = null
      store.setCurrentPageId('SIGNUP_FORM_BUILDER')
      await router.push({ name: 'SIGNUP_FORM_BUILDER' })
    }

    async function createForm() {
      store.setCurrentPageId('SIGNUP_FORM_BUILDER')
      await router.push({ name: 'SIGNUP_FORM_BUILDER' })
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
      createForm,
      goBackDashboard
    }
  }
}
</script>