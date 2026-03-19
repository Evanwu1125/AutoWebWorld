<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <header class="bg-white shadow-sm z-10">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
        <h1 class="text-2xl font-bold text-[#005DAA]">Appointments</h1>
        <button id="back-dashboard" @click="handleBack" class="text-gray-600 hover:text-gray-900">
          Back
        </button>
      </div>
    </header>

    <main class="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
      <div class="bg-white p-4 rounded-lg shadow mb-6 space-y-4 md:space-y-0 md:flex md:items-center md:space-x-4">
        <!-- Search -->
        <div class="flex-1 relative">
           <input
             id="appts-search-input"
             type="text"
             placeholder="Search provider..."
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
             id="filter-appt-virtual-checkbox"
             type="checkbox"
             v-model="filterVirtual"
             @change="handleFilterChange"
             class="h-4 w-4 text-[#005DAA] focus:ring-[#005DAA] border-gray-300 rounded"
           />
           <label for="filter-appt-virtual-checkbox" class="ml-2 block text-sm text-gray-900">
             Virtual Only
           </label>
        </div>
      </div>

      <!-- List -->
      <div id="appts-list-container" class="space-y-4">
        <div 
          v-for="appt in filteredAppointments" 
          :key="appt.id"
          class="bg-white rounded-lg shadow hover:shadow-md transition-shadow duration-200"
          :class="{
             'ring-2 ring-green-500': appt.id === matchedId,
             'ring-2 ring-blue-500': appt.id === store.appts_list_viewport_anchor_id
          }"
        >
          <div 
             :id="appt.id === matchedId ? 'appts-list-item-matched' : (isFiltered ? 'appts-list-item-filtered' : 'appts-list-item-visible')"
             :class="`data-id-${appt.id} p-6 flex items-start space-x-4 cursor-pointer`"
             @click="handleSelectAppt(appt)"
          >
             <div class="h-16 w-16 bg-gray-200 rounded-full flex-shrink-0 overflow-hidden">
                <img :src="appt.image" class="h-full w-full object-cover" />
             </div>
             
             <div class="flex-1 min-w-0">
               <div class="flex justify-between">
                 <h3 class="text-lg font-bold text-gray-900">{{ appt.provider }}</h3>
                 <span 
                   class="px-2 py-1 text-xs font-semibold rounded-full"
                   :class="appt.status === 'Upcoming' ? 'bg-blue-100 text-blue-800' : 'bg-gray-100 text-gray-800'"
                 >
                   {{ appt.status }}
                 </span>
               </div>
               <p class="text-gray-500">{{ appt.date }} at {{ appt.time }}</p>
               <p class="text-sm text-[#009CDE] mt-1">{{ appt.type }} Visit</p>
             </div>
             
             <div class="self-center">
                <svg class="h-6 w-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path></svg>
             </div>
          </div>
        </div>

        <div v-if="filteredAppointments.length === 0" class="text-center py-12">
           <p class="text-gray-500">No appointments found.</p>
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
  name: 'APPOINTMENTS_LIST',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const searchQuery = ref('')
    const filterVirtual = ref(false)
    const matchedId = ref(null)

    const filteredAppointments = computed(() => {
      let result = dataStore.appointments

      if (filterVirtual.value) {
        result = result.filter(a => a.type === 'Virtual')
      }

      return result
    })

    const isFiltered = computed(() => filterVirtual.value)

    const handleSearch = () => {
      // ACT_APPTS_SEARCH
      const match = dataStore.appointments.find(a => a.provider.toLowerCase().includes(searchQuery.value.toLowerCase()))
      if (match) {
        store.matched_appointment_id = match.id
        store.appts_list_has_searched = true
        matchedId.value = match.id
      } else {
        matchedId.value = null
      }
    }

    const handleFilterChange = () => {
      // ACT_APPTS_FILTER_TYPE
      store.appts_list_filters_applied = true
    }

    const handleSelectAppt = async (appt) => {
      // ACT_APPTS_OPEN_MATCHED, ACT_APPTS_OPEN_ANY, ACT_APPTS_OPEN_FILTERED
      store.selected_appointment_id = appt.id
      
      store.appts_list_has_searched = false
      store.appts_list_viewport_anchor_id = null
      store.appts_list_filters_applied = false

      store.setCurrentPageId('APPOINTMENT_DETAIL')
      await router.push({ name: 'APPOINTMENT_DETAIL' })
    }

    const handleBack = async () => {
      // ACT_APPTS_BACK_DASH
      store.setCurrentPageId('DASHBOARD')
      await router.push({ name: 'DASHBOARD' })
    }

    return {
      store,
      searchQuery,
      filterVirtual,
      matchedId,
      filteredAppointments,
      isFiltered,
      handleSearch,
      handleFilterChange,
      handleSelectAppt,
      handleBack
    }
  }
}
</script>