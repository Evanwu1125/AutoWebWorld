<template>
  <div class="min-h-screen bg-slate-50 flex flex-col font-inter text-slate-900">
    <!-- Header -->
    <header class="bg-white shadow-sm z-20">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
        <h1 class="text-2xl font-bold text-slate-900">Tickets</h1>
        <div class="flex space-x-4">
            <button id="btn-new-ticket" @click="handleNewTicket" class="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-md shadow-sm transition-colors duration-200 flex items-center">
              <span class="mr-2">＋</span> New Ticket
            </button>
            <button id="back-home" @click="handleBackHome" class="bg-white border border-slate-300 text-slate-700 hover:bg-slate-50 font-medium py-2 px-4 rounded-md shadow-sm transition-colors duration-200">
              Home
            </button>
        </div>
      </div>
    </header>

    <main class="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
      <!-- Toolbar: Search, Sort, Filters -->
      <div class="bg-white p-4 rounded-lg shadow-sm mb-6 space-y-4 lg:space-y-0 lg:flex lg:items-center lg:justify-between">
         <!-- Search -->
         <div class="flex-1 max-w-lg">
            <label for="tickets-search-input" class="sr-only">Search</label>
            <div class="relative rounded-md shadow-sm">
              <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <span class="text-slate-400">🔍</span>
              </div>
              <input type="text" 
                     id="tickets-search-input"
                     v-model="searchQuery"
                     @keypress.enter="handleSearch"
                     class="focus:ring-blue-500 focus:border-blue-500 block w-full pl-10 sm:text-sm border-slate-300 rounded-md py-2" 
                     placeholder="Search tickets...">
            </div>
         </div>

         <!-- Sort -->
         <div class="relative ml-4">
             <button id="tickets-sort-dropdown" @click="toggleSortDropdown" class="bg-white border border-slate-300 text-slate-700 py-2 px-4 rounded-md shadow-sm text-sm font-medium hover:bg-slate-50 flex items-center">
               Sort by: {{ sortLabel }} <span class="ml-2">▼</span>
             </button>
             <div v-if="sortDropdownOpen" class="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg py-1 border border-slate-100 z-50">
                <div id="sort-option-newest-desc" @click="handleSort('newest')" class="block px-4 py-2 text-sm text-slate-700 hover:bg-slate-50 cursor-pointer">Newest</div>
                <div id="sort-option-oldest" @click="handleSort('oldest')" class="block px-4 py-2 text-sm text-slate-700 hover:bg-slate-50 cursor-pointer">Oldest</div>
                <div id="sort-option-priority" @click="handleSort('priority')" class="block px-4 py-2 text-sm text-slate-700 hover:bg-slate-50 cursor-pointer">Priority</div>
             </div>
         </div>
      </div>

      <!-- Filters Section -->
      <div class="bg-white p-4 rounded-lg shadow-sm mb-6">
        <h3 class="text-sm font-medium text-slate-500 mb-3 uppercase tracking-wider">Filters</h3>
        <div class="flex flex-wrap gap-6 items-start">
           <!-- Status Checkboxes -->
           <div class="space-y-2">
             <label class="text-sm font-medium text-slate-700">Status</label>
             <div class="flex items-center space-x-4">
               <label class="inline-flex items-center">
                 <input type="checkbox" id="filter-status-open-checkbox" v-model="filterStatusOpen" @change="applyFilters" class="form-checkbox h-4 w-4 text-blue-600 rounded border-slate-300 focus:ring-blue-500">
                 <span class="ml-2 text-sm text-slate-600">Open</span>
               </label>
               <!-- Add more if needed, FSM only specifies Open checkbox action explicitly -->
             </div>
           </div>
           
           <!-- Priority Checkboxes -->
           <div class="space-y-2">
             <label class="text-sm font-medium text-slate-700">Priority</label>
             <div class="flex items-center space-x-4">
                <label class="inline-flex items-center">
                 <input type="checkbox" id="filter-priority-high-checkbox" v-model="filterPriorityHigh" @change="applyFilters" class="form-checkbox h-4 w-4 text-blue-600 rounded border-slate-300 focus:ring-blue-500">
                 <span class="ml-2 text-sm text-slate-600">High</span>
               </label>
             </div>
           </div>

           <!-- Agent Load Slider (Mock) -->
           <div class="space-y-2 flex-1 min-w-[200px]">
              <div class="flex justify-between">
                <label class="text-sm font-medium text-slate-700">Agent Load</label>
                <span class="text-xs text-slate-500">{{ agentLoad }}%</span>
              </div>
              <input type="range" 
                     id="filter-agent-load-slider" 
                     min="0" max="100" step="1"
                     v-model="agentLoad" 
                     @input="applyFilters"
                     class="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600">
           </div>
        </div>
      </div>

      <!-- Tickets List Table -->
      <div class="bg-white shadow overflow-hidden sm:rounded-md" id="tickets-table">
        <ul role="list" class="divide-y divide-slate-200">
          <li v-for="ticket in filteredTickets" :key="ticket.id" class="hover:bg-slate-50 transition-colors duration-150">
             <div 
                  :class="[
                    'block px-4 py-4 sm:px-6 cursor-pointer',
                    `data-id-${ticket.id}`,
                    // Classes for FSM selection logic
                    isMatched(ticket) ? 'row-matched' : '',
                    isFilteredFirst(ticket) ? 'row-filtered-first' : '',
                    'row-visible' // All rendered rows are visible
                  ]"
                  @click="handleOpenTicket(ticket)"
             >
                <div class="flex items-center justify-between">
                  <div class="flex items-center truncate">
                     <p class="text-sm font-medium text-blue-600 truncate mr-4">#{{ ticket.id }}</p>
                     <p class="text-sm text-slate-900 truncate font-semibold">{{ ticket.subject }}</p>
                  </div>
                  <div class="ml-2 flex-shrink-0 flex">
                    <span :class="[
                      'px-2 inline-flex text-xs leading-5 font-semibold rounded-full',
                      statusColor(ticket.status)
                    ]">
                      {{ ticket.status }}
                    </span>
                  </div>
                </div>
                <div class="mt-2 sm:flex sm:justify-between">
                  <div class="sm:flex">
                    <p class="flex items-center text-sm text-slate-500">
                      <span class="truncate">{{ ticket.description }}</span>
                    </p>
                  </div>
                  <div class="mt-2 flex items-center text-sm text-slate-500 sm:mt-0">
                    <!-- Priority Badge -->
                    <span :class="[
                        'mr-4 px-2 py-0.5 rounded text-xs border',
                        priorityColor(ticket.priority)
                    ]">
                        {{ ticket.priority }}
                    </span>
                    <p>
                      Created <time :datetime="ticket.created_at">{{ formatDate(ticket.created_at) }}</time>
                    </p>
                    <img :src="ticket.image" class="h-6 w-6 rounded-full ml-3 border border-slate-200" alt="Thumbnail">
                  </div>
                </div>
             </div>
          </li>
          
          <!-- Empty State -->
          <li v-if="filteredTickets.length === 0" class="px-4 py-12 text-center text-slate-500">
             <div class="mx-auto h-12 w-12 text-slate-300 text-4xl mb-4">📭</div>
             <p>No tickets found matching your criteria.</p>
          </li>
        </ul>
      </div>

    </main>
    
    <!-- Location Permission Modal -->
    <div v-if="showLocationModal" class="fixed inset-0 bg-slate-900 bg-opacity-50 z-[9999] flex items-center justify-center">
       <div class="bg-white rounded-lg p-6 max-w-sm w-full shadow-xl">
          <h3 class="text-lg font-medium text-slate-900 mb-2">Location Permission Required</h3>
          <p class="text-sm text-slate-500 mb-4">This app needs access to your location to provide better service.</p>
          <button id="permission-location-allow" @click="handleGrantLocation" class="w-full bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 transition">Allow</button>
       </div>
    </div>

  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'
import _ from 'lodash-es'

export default {
  name: 'TICKETS_LIST',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const searchQuery = ref('')
    const filterStatusOpen = ref(false)
    const filterPriorityHigh = ref(false)
    const agentLoad = ref(50)
    const sortOption = ref('')
    const sortDropdownOpen = ref(false)

    // Derived state for FSM logic
    const showLocationModal = computed(() => !signatureStore.location_permission_granted)
    
    // Filtering Logic
    const filteredTickets = computed(() => {
        let result = [...dataStore.tickets]
        
        // Search
        if (searchQuery.value) {
            const q = searchQuery.value.toLowerCase()
            result = result.filter(t => 
                t.subject.toLowerCase().includes(q) || 
                t.description.toLowerCase().includes(q) ||
                t.id.toLowerCase().includes(q)
            )
        }

        // Filters
        if (filterStatusOpen.value) {
            result = result.filter(t => t.status === 'Open')
        }
        if (filterPriorityHigh.value) {
            result = result.filter(t => t.priority === 'High')
        }
        // Agent Load Slider logic (simulated: filter based on arbitrary 'load' or just show subset)
        // For FSM demo, we'll assume slider > 80 filters heavily
        if (agentLoad.value > 80) {
             // Mock behavior: only show first 5 items when load is high
             result = result.slice(0, 5)
        }

        // Sorting
        if (sortOption.value) {
            if (sortOption.value === 'newest') {
                result.sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
            } else if (sortOption.value === 'oldest') {
                result.sort((a, b) => new Date(a.created_at) - new Date(b.created_at))
            } else if (sortOption.value === 'priority') {
                const pMap = { 'High': 3, 'Medium': 2, 'Low': 1 }
                result.sort((a, b) => (pMap[b.priority] || 0) - (pMap[a.priority] || 0))
            }
        }

        return result
    })

    const isMatched = (ticket) => {
        if (!signatureStore.tickets_list_has_searched) return false
        // Basic match logic: if it's the first result of a search
        if (filteredTickets.value.length > 0 && filteredTickets.value[0].id === ticket.id) return true
        return false
    }

    const isFilteredFirst = (ticket) => {
        if (!signatureStore.tickets_list_filters_applied) return false
        if (filteredTickets.value.length > 0 && filteredTickets.value[0].id === ticket.id) return true
        return false
    }

    // Actions
    const handleNewTicket = async () => {
        signatureStore.setCurrentPageId('NEW_TICKET_FORM')
        await router.push({ name: 'NEW_TICKET_FORM' })
    }

    const handleBackHome = async () => {
        signatureStore.setCurrentPageId('HOME')
        await router.push({ name: 'HOME' })
    }

    const handleSearch = () => {
        signatureStore.tickets_list_has_searched = true
        signatureStore.matched_ticket_id = filteredTickets.value.length > 0 ? filteredTickets.value[0].id : null
    }

    const toggleSortDropdown = () => {
        sortDropdownOpen.value = !sortDropdownOpen.value
    }

    const handleSort = (option) => {
        sortOption.value = option
        signatureStore.tickets_list_filters_applied = true
        sortDropdownOpen.value = false
    }

    const applyFilters = () => {
        signatureStore.tickets_list_filters_applied = true
    }

    const handleGrantLocation = () => {
        signatureStore.location_permission_granted = true
    }

    const handleOpenTicket = async (ticket) => {
        signatureStore.selected_ticket_id = ticket.id
        // Clear flags based on context (handled in FSM effects, here we simulate)
        if (signatureStore.tickets_list_filters_applied) signatureStore.tickets_list_filters_applied = null
        if (signatureStore.tickets_list_has_searched) signatureStore.tickets_list_has_searched = null
        if (signatureStore.tickets_list_viewport_anchor_id) signatureStore.tickets_list_viewport_anchor_id = null
        
        signatureStore.setCurrentPageId('TICKET_DETAIL')
        await router.push({ name: 'TICKET_DETAIL', params: { id: ticket.id } })
    }

    // Helpers
    const statusColor = (status) => {
        if (status === 'Open') return 'bg-green-100 text-green-800'
        if (status === 'Pending') return 'bg-yellow-100 text-yellow-800'
        if (status === 'Resolved') return 'bg-gray-100 text-gray-800'
        return 'bg-gray-100 text-gray-800'
    }

    const priorityColor = (priority) => {
        if (priority === 'High') return 'border-red-200 text-red-700 bg-red-50'
        if (priority === 'Medium') return 'border-yellow-200 text-yellow-700 bg-yellow-50'
        return 'border-blue-200 text-blue-700 bg-blue-50'
    }

    const formatDate = (dateStr) => {
        return new Date(dateStr).toLocaleDateString()
    }

    return {
        searchQuery,
        filterStatusOpen,
        filterPriorityHigh,
        agentLoad,
        sortDropdownOpen,
        sortLabel: computed(() => sortOption.value ? sortOption.value.charAt(0).toUpperCase() + sortOption.value.slice(1) : 'Default'),
        filteredTickets,
        showLocationModal,
        signatureStore,
        handleNewTicket,
        handleBackHome,
        handleSearch,
        toggleSortDropdown,
        handleSort,
        applyFilters,
        handleGrantLocation,
        handleOpenTicket,
        statusColor,
        priorityColor,
        formatDate,
        isMatched,
        isFilteredFirst
    }
  }
}
</script>