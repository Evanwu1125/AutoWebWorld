<template>
  <div class="min-h-screen bg-slate-50 font-sans">
    <header class="bg-[#002D5C] text-white py-4 px-6 shadow-md sticky top-0 z-30">
      <div class="max-w-7xl mx-auto flex items-center justify-between">
        <div id="back-multi-search" @click="goBack" class="flex items-center gap-2 cursor-pointer hover:text-blue-200 transition-colors">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"/></svg>
          <span class="font-medium">Modify Trip</span>
        </div>
        <div class="font-bold text-xl">Multi-City Results</div>
        <div class="w-24"></div>
      </div>
    </header>

    <div class="max-w-7xl mx-auto px-6 py-8 flex gap-8">
      <!-- Sidebar -->
      <aside class="w-64 shrink-0 hidden md:block space-y-6">
        <div class="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
          <h3 class="font-bold text-gray-900 mb-4 border-b border-gray-100 pb-2">Stops</h3>
           <label class="flex items-center gap-3 cursor-pointer group">
            <div class="relative flex items-center">
              <input 
                id="multi-filter-nonstop"
                type="checkbox" 
                @change="handleFilterNonstop"
                class="peer h-5 w-5 cursor-pointer appearance-none rounded border border-gray-300 shadow-sm transition-all checked:border-blue-600 checked:bg-blue-600 hover:border-blue-400 focus:ring-2 focus:ring-blue-200" 
              />
              <svg class="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-3.5 h-3.5 pointer-events-none opacity-0 peer-checked:opacity-100 text-white" viewBox="0 0 14 14" fill="none">
                <path d="M3 8L6 11L11 3.5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
              </svg>
            </div>
            <span class="text-gray-700 group-hover:text-blue-600 transition-colors">Direct flights only</span>
          </label>
        </div>
      </aside>

      <!-- List -->
      <div class="flex-1 space-y-6">
        <div id="multi-results-list" @scroll="handleScroll" class="space-y-4">
          <div 
            v-for="flight in filteredFlights" 
            :key="flight.id"
            :class="[
              `data-id-${flight.id}`,
              'bg-white rounded-2xl shadow-sm hover:shadow-md transition-all border border-gray-100 p-6 cursor-pointer flex flex-col md:flex-row gap-6',
              isFiltered ? 'multi-option-filtered' : 'multi-option-visible'
            ]"
            @click="openFlight(flight.id)"
          >
             <!-- Airline Info -->
            <div class="w-full md:w-1/4 flex items-center gap-4">
              <div class="w-12 h-12 rounded-full bg-gray-50 overflow-hidden border border-gray-100 flex-shrink-0">
                <img :src="flight.image" :alt="flight.airline" class="w-full h-full object-cover" />
              </div>
              <div class="font-bold text-gray-900">{{ flight.airline }}</div>
            </div>

            <!-- Legs Info -->
            <div class="flex-1 flex flex-col justify-center space-y-2">
              <div v-for="(leg, idx) in flight.legs" :key="idx" class="flex items-center gap-2 text-sm text-gray-600">
                <svg class="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>
                <span>{{ leg }}</span>
              </div>
              <div class="text-xs text-gray-400 mt-2">{{ flight.total_duration }} • {{ flight.stops }} Stop(s) total</div>
            </div>

             <!-- Price -->
            <div class="w-full md:w-1/4 border-t md:border-t-0 md:border-l border-gray-100 md:pl-6 pt-4 md:pt-0 flex flex-row md:flex-col justify-between md:justify-center items-center gap-2">
              <div class="text-2xl font-bold text-gray-900">£{{ flight.price }}</div>
              <div class="text-xs text-gray-400">Total Price</div>
              <button class="w-full mt-2 py-2 bg-blue-600 text-white font-bold rounded-lg hover:bg-blue-700 transition-colors hidden md:block">
                Select
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'MULTI_CITY_RESULTS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const isFiltered = computed(() => store.multi_results_filters_applied)

    const filteredFlights = computed(() => {
      // Mock filter logic
      // Ideally check store.multi_results_filters_applied
      return dataStore.multiFlights
    })

    const handleFilterNonstop = () => {
      store.multi_results_filters_applied = true
    }

    const handleScroll = (e) => {
      if (filteredFlights.value.length > 0) {
        store.multi_results_viewport_anchor_id = filteredFlights.value[0].id
      }
    }

    const openFlight = async (id) => {
      // FSM logic: check precondition viewport_anchor_id > 0 if scrolling used
      // Or simply navigation.
      // But FSM actions require viewport_anchor_id for 'MULTI_RESULTS_OPEN_OPTION' if we strictly follow preconditions.
      // However, if user clicks, it implies selection. We'll set the ID.
      store.multi_results_viewport_anchor_id = id // satisfy precondition if strict
      store.currentPageId = 'MULTI_CITY_INTRO'
      await router.push({ name: 'MULTI_CITY_INTRO' })
    }

    const goBack = async () => {
      store.currentPageId = 'MULTI_CITY_SEARCH'
      await router.push({ name: 'MULTI_CITY_SEARCH' })
    }

    return {
      filteredFlights,
      isFiltered,
      handleFilterNonstop,
      handleScroll,
      openFlight,
      goBack
    }
  }
}
</script>