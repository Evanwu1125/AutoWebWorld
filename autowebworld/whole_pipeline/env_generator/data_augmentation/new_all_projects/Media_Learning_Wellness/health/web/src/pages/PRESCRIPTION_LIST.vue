<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <header class="bg-white shadow-sm z-10">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
        <h1 class="text-2xl font-bold text-[#005DAA]">My Prescriptions</h1>
        <button id="back-dashboard" @click="handleBack" class="text-gray-600 hover:text-gray-900">
          Back
        </button>
      </div>
    </header>

    <main class="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
      <!-- Search & Filter -->
      <div class="bg-white p-4 rounded-lg shadow mb-6 space-y-4 md:space-y-0 md:flex md:items-center md:space-x-4">
        <!-- Search -->
        <div class="flex-1 relative">
           <input
             id="rx-search-input"
             type="text"
             placeholder="Search prescriptions..."
             v-model="searchQuery"
             @keyup.enter="handleSearch"
             class="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:ring-[#009CDE] focus:border-[#009CDE]"
           />
           <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
             <svg class="h-5 w-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
           </div>
        </div>

        <!-- Filter -->
        <div class="flex items-center">
           <input
             id="filter-rx-active-checkbox"
             type="checkbox"
             v-model="filterActive"
             @change="handleFilterChange"
             class="h-4 w-4 text-[#005DAA] focus:ring-[#005DAA] border-gray-300 rounded"
           />
           <label for="filter-rx-active-checkbox" class="ml-2 block text-sm text-gray-900">
             Active Only
           </label>
        </div>
      </div>

      <!-- List -->
      <div id="rx-list-container" class="space-y-4">
        <div 
          v-for="rx in filteredPrescriptions" 
          :key="rx.id"
          class="bg-white rounded-lg shadow hover:shadow-md transition-shadow duration-200"
          :class="{
             'ring-2 ring-green-500': rx.id === matchedId,
             'ring-2 ring-blue-500': rx.id === store.prescription_list_viewport_anchor_id
          }"
        >
          <div 
             :id="rx.id === matchedId ? 'rx-list-item-matched' : (isFiltered ? 'rx-list-item-filtered' : 'rx-list-item-visible')"
             :class="`data-id-${rx.id} p-6 flex items-center space-x-4 cursor-pointer`"
             @click="handleSelectRx(rx)"
          >
             <div class="h-12 w-12 bg-blue-50 rounded-lg flex items-center justify-center text-[#005DAA]">
               <svg class="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.384-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"></path></svg>
             </div>
             
             <div class="flex-1">
               <div class="flex items-center justify-between">
                 <h3 class="text-lg font-bold text-gray-900">{{ rx.name }}</h3>
                 <span 
                   class="px-2 py-1 text-xs font-semibold rounded-full"
                   :class="rx.status === 'Active' ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'"
                 >
                   {{ rx.status }}
                 </span>
               </div>
               <p class="text-sm text-gray-500">{{ rx.dosage }} • {{ rx.supply }} supply</p>
             </div>
             
             <div class="self-center">
                <svg class="h-6 w-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path></svg>
             </div>
          </div>
        </div>

        <div v-if="filteredPrescriptions.length === 0" class="text-center py-12">
           <p class="text-gray-500">No prescriptions found.</p>
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
  name: 'PRESCRIPTION_LIST',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const searchQuery = ref('')
    const filterActive = ref(false)
    const matchedId = ref(null)

    const filteredPrescriptions = computed(() => {
      let result = dataStore.prescriptions

      if (filterActive.value) {
        result = result.filter(r => r.status === 'Active')
      }

      return result
    })

    const isFiltered = computed(() => filterActive.value)

    const handleSearch = () => {
      // ACT_RX_SEARCH
      const match = dataStore.prescriptions.find(r => r.name.toLowerCase().includes(searchQuery.value.toLowerCase()))
      if (match) {
        store.matched_prescription_id = match.id
        store.prescription_list_has_searched = true
        matchedId.value = match.id
      } else {
        matchedId.value = null
      }
    }

    const handleFilterChange = () => {
      // ACT_RX_FILTER_STATUS
      store.prescription_list_filters_applied = true
    }

    const handleSelectRx = async (rx) => {
      // ACT_RX_OPEN_MATCHED, ACT_RX_OPEN_ANY, ACT_RX_OPEN_FILTERED
      store.selected_prescription_id = rx.id
      
      store.prescription_list_has_searched = false
      store.prescription_list_viewport_anchor_id = null
      store.prescription_list_filters_applied = false

      store.setCurrentPageId('PRESCRIPTION_DETAIL')
      await router.push({ name: 'PRESCRIPTION_DETAIL' })
    }

    const handleBack = async () => {
      // ACT_RX_BACK_DASH
      store.setCurrentPageId('DASHBOARD')
      await router.push({ name: 'DASHBOARD' })
    }

    return {
      store,
      searchQuery,
      filterActive,
      matchedId,
      filteredPrescriptions,
      isFiltered,
      handleSearch,
      handleFilterChange,
      handleSelectRx,
      handleBack
    }
  }
}
</script>