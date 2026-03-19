<template>
  <div class="min-h-screen bg-slate-50 font-inter text-slate-900 flex flex-col">
    <!-- Header -->
    <header class="bg-white shadow-sm z-20">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
         <h1 class="text-2xl font-bold text-slate-900">Dashboard</h1>
         <button id="dashboard-back-home" @click="handleBackHome" class="bg-white border border-slate-300 text-slate-700 hover:bg-slate-50 font-medium py-2 px-4 rounded-md shadow-sm transition-colors duration-200">
            Home
         </button>
      </div>
    </header>

    <main class="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
      <!-- Toolbar: Filters/Sort -->
      <div class="bg-white p-4 rounded-lg shadow-sm mb-6 space-y-4 lg:space-y-0 lg:flex lg:items-center lg:justify-between">
         <!-- Channel Filter -->
         <div class="flex items-center space-x-4">
             <label class="inline-flex items-center">
                 <input type="checkbox" id="filter-channel-email" v-model="filterChannelEmail" @change="applyFilters" class="form-checkbox h-4 w-4 text-blue-600 rounded border-slate-300 focus:ring-blue-500">
                 <span class="ml-2 text-sm text-slate-600">Email Channel</span>
             </label>
         </div>

         <!-- Satisfaction Slider -->
         <div class="flex-1 max-w-xs mx-4">
            <div class="flex justify-between">
                <label class="text-sm font-medium text-slate-700">Min Satisfaction</label>
                <span class="text-xs text-slate-500">{{ satisfaction }}%</span>
            </div>
            <input type="range" 
                   id="filter-satisfaction-slider" 
                   min="0" max="100" step="1"
                   v-model="satisfaction" 
                   @input="applyFilters"
                   class="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600">
         </div>

         <!-- Sort Widget -->
         <div class="relative">
             <button id="dashboard-sort-dropdown" @click="toggleSortDropdown" class="bg-white border border-slate-300 text-slate-700 py-2 px-4 rounded-md shadow-sm text-sm font-medium hover:bg-slate-50 flex items-center">
               Sort Widgets by <span class="ml-2">▼</span>
             </button>
             <div v-if="sortDropdownOpen" class="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg py-1 border border-slate-100 z-50">
                <div id="dashboard-sort-volume" @click="handleSort('volume')" class="block px-4 py-2 text-sm text-slate-700 hover:bg-slate-50 cursor-pointer">Volume</div>
                <div id="dashboard-sort-satisfaction" @click="handleSort('satisfaction')" class="block px-4 py-2 text-sm text-slate-700 hover:bg-slate-50 cursor-pointer">Satisfaction</div>
             </div>
         </div>
      </div>

      <!-- Dashboard Widgets Grid -->
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
         <!-- Unresolved Tickets Widget -->
         <div id="dashboard-widget-tickets" 
              @click="handleOpenFilteredTickets"
              class="bg-white p-6 rounded-lg shadow-sm border border-slate-200 hover:shadow-md transition-shadow cursor-pointer group">
            <h3 class="text-sm font-medium text-slate-500 uppercase tracking-wider mb-2">Unresolved Tickets</h3>
            <div class="flex items-baseline">
               <span class="text-3xl font-bold text-slate-900">{{ unresolvedCount }}</span>
               <span class="ml-2 text-sm text-red-600 group-hover:text-red-700 font-medium">Require Attention</span>
            </div>
            <div class="mt-4 h-2 bg-slate-100 rounded-full overflow-hidden">
               <div class="h-full bg-blue-500 w-3/4"></div>
            </div>
         </div>

         <!-- Satisfaction Widget -->
         <div class="bg-white p-6 rounded-lg shadow-sm border border-slate-200">
            <h3 class="text-sm font-medium text-slate-500 uppercase tracking-wider mb-2">Cust. Satisfaction</h3>
            <div class="flex items-baseline">
               <span class="text-3xl font-bold text-slate-900">94%</span>
               <span class="ml-2 text-sm text-green-600 font-medium">↑ 2% vs last week</span>
            </div>
            <div class="mt-4 h-2 bg-slate-100 rounded-full overflow-hidden">
               <div class="h-full bg-green-500 w-[94%]"></div>
            </div>
         </div>

         <!-- Response Time Widget -->
         <div class="bg-white p-6 rounded-lg shadow-sm border border-slate-200">
            <h3 class="text-sm font-medium text-slate-500 uppercase tracking-wider mb-2">Avg Response Time</h3>
            <div class="flex items-baseline">
               <span class="text-3xl font-bold text-slate-900">2h 15m</span>
               <span class="ml-2 text-sm text-slate-500">Target: &lt; 4h</span>
            </div>
            <div class="mt-4 h-2 bg-slate-100 rounded-full overflow-hidden">
               <div class="h-full bg-purple-500 w-1/2"></div>
            </div>
         </div>
      </div>
      
      <!-- Chart Placeholder Area -->
      <div class="mt-8 bg-white p-6 rounded-lg shadow-sm border border-slate-200 min-h-[300px] flex items-center justify-center bg-slate-50">
         <p class="text-slate-400 font-medium">Chart Visualization Area (Placeholder)</p>
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
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'DASHBOARD',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const filterChannelEmail = ref(false)
    const satisfaction = ref(80)
    const sortDropdownOpen = ref(false)

    // Derived state for FSM logic
    const showLocationModal = computed(() => !signatureStore.location_permission_granted)
    
    // Mock metric
    const unresolvedCount = computed(() => {
        return dataStore.tickets.filter(t => t.status === 'Open' || t.status === 'Pending').length
    })

    const handleBackHome = async () => {
        signatureStore.setCurrentPageId('HOME')
        await router.push({ name: 'HOME' })
    }

    const applyFilters = () => {
        signatureStore.dashboard_filters_applied = true
    }

    const toggleSortDropdown = () => sortDropdownOpen.value = !sortDropdownOpen.value

    const handleSort = (val) => {
        // Just triggers filter applied in FSM
        signatureStore.dashboard_filters_applied = true
        sortDropdownOpen.value = false
    }

    const handleOpenFilteredTickets = async () => {
        if (!signatureStore.dashboard_filters_applied) return // FSM condition: filters applied eq true. But for UX, we might want to navigate anyway? FSM says PRECONDITION filters_applied=true.
        
        // Wait, FSM ACT_DASHBOARD_OPEN_FILTERED_TICKETS precondition is: dashboard_filters_applied == true.
        // If user hasn't touched filters, they can't click?
        // To make it usable, let's auto-apply filter if they click the widget, or assume default state satisfies it if we set it.
        // Or strictly follow FSM: Button does nothing unless filters applied.
        // Let's strictly follow FSM but maybe initialize it or rely on user interaction.
        // Better UX: clicking widget sets the flag if not set? No, FSM logic is strict.
        
        // Let's assume user MUST interact with filter first per FSM.
        if (signatureStore.dashboard_filters_applied === true) {
             // Clear flag effect
             signatureStore.dashboard_filters_applied = null
             signatureStore.setCurrentPageId('TICKETS_LIST')
             await router.push({ name: 'TICKETS_LIST' })
        } else {
             alert("Please apply a filter or sort first (per FSM requirement).")
        }
    }

    const handleGrantLocation = () => {
        signatureStore.location_permission_granted = true
    }

    return {
        filterChannelEmail,
        satisfaction,
        sortDropdownOpen,
        showLocationModal,
        unresolvedCount,
        handleBackHome,
        applyFilters,
        toggleSortDropdown,
        handleSort,
        handleOpenFilteredTickets,
        handleGrantLocation
    }
  }
}
</script>