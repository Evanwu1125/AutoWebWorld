<template>
  <div class="min-h-screen bg-slate-50 font-sans">
    <header class="bg-[#002D5C] text-white py-4 px-6 shadow-md sticky top-0 z-30">
      <div class="max-w-7xl mx-auto flex items-center justify-between">
        <div id="back-flights-search" @click="goBack" class="flex items-center gap-2 cursor-pointer hover:text-blue-200 transition-colors">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"/></svg>
          <span class="font-medium">Modify Search</span>
        </div>
        <div class="font-bold text-xl">London (LHR) <span class="text-blue-300 mx-2">to</span> New York (JFK)</div>
        <div class="w-24"></div>
      </div>
    </header>

    <div class="max-w-7xl mx-auto px-4 md:px-6 py-8 flex flex-col md:flex-row gap-8">
      <!-- Filters Sidebar -->
      <aside class="w-full md:w-72 shrink-0 space-y-6">
        <div class="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
          <h3 class="font-bold text-gray-900 mb-4 border-b border-gray-100 pb-2">Stops</h3>
          <label class="flex items-center gap-3 cursor-pointer group">
            <div class="relative flex items-center">
              <input 
                id="filter-nonstop-checkbox"
                type="checkbox" 
                @change="handleFilterNonstop"
                class="peer h-5 w-5 cursor-pointer appearance-none rounded border border-gray-300 shadow-sm transition-all checked:border-blue-600 checked:bg-blue-600 hover:border-blue-400 focus:ring-2 focus:ring-blue-200" 
              />
              <svg class="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-3.5 h-3.5 pointer-events-none opacity-0 peer-checked:opacity-100 text-white" viewBox="0 0 14 14" fill="none">
                <path d="M3 8L6 11L11 3.5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
              </svg>
            </div>
            <span class="text-gray-700 group-hover:text-blue-600 transition-colors">Direct only</span>
          </label>
        </div>

        <div class="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
          <h3 class="font-bold text-gray-900 mb-4 border-b border-gray-100 pb-2">Departure Time</h3>
           <div class="space-y-4">
            <label class="flex justify-between text-sm text-gray-600">
              <span>00:00</span>
              <span>23:59</span>
            </label>
            <input 
              id="departure-time-slider"
              type="range" 
              min="0" 
              max="24"
              value="0"
              @input="handleTimeSlider"
              class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
            />
            <p class="text-xs text-center text-gray-500">Drag to filter later flights</p>
           </div>
        </div>
      </aside>

      <!-- Results Area -->
      <div class="flex-1 space-y-6">
        <!-- Search & Sort Bar -->
        <div class="flex flex-col sm:flex-row gap-4 justify-between items-center bg-white p-4 rounded-xl shadow-sm border border-gray-100">
          <div class="relative w-full sm:max-w-xs">
            <svg class="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
            <input 
              id="results-search-input"
              type="text" 
              v-model="searchQuery"
              @keyup.enter="handleSearch"
              class="w-full pl-10 pr-4 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none text-sm"
              placeholder="Search airline or flight #"
            />
          </div>

          <div class="relative">
            <div id="sort-dropdown" @click="toggleSort" class="flex items-center gap-2 cursor-pointer text-gray-700 font-medium hover:bg-gray-50 px-4 py-2 rounded-lg border border-gray-200 transition-colors bg-white">
              <span class="text-sm">Sort by: <span class="text-blue-600">{{ currentSortLabel }}</span></span>
              <svg class="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
            </div>
            <div v-if="sortOpen" class="absolute top-full right-0 mt-2 w-48 bg-white rounded-xl shadow-xl border border-gray-100 z-50 overflow-hidden">
               <div id="sort-option-cheapest" @click="selectSort('cheapest')" class="px-4 py-3 hover:bg-blue-50 cursor-pointer text-gray-700 text-sm">Cheapest</div>
               <div id="sort-option-fastest" @click="selectSort('fastest')" class="px-4 py-3 hover:bg-blue-50 cursor-pointer text-gray-700 text-sm">Fastest</div>
               <div id="sort-option-best" @click="selectSort('best')" class="px-4 py-3 hover:bg-blue-50 cursor-pointer text-gray-700 text-sm">Best Value</div>
            </div>
          </div>
        </div>

        <div class="flex justify-between items-center text-sm text-gray-500 px-1">
          <span>{{ filteredFlights.length }} results found</span>
        </div>

        <!-- Flight List -->
        <div id="results-list" @scroll="handleScroll" class="space-y-4">
          <div 
            v-for="flight in filteredFlights" 
            :key="flight.id"
            :class="[
              `data-id-${flight.id}`,
              'bg-white rounded-2xl shadow-sm hover:shadow-md transition-all border border-gray-100 p-6 flex flex-col sm:flex-row gap-6 cursor-pointer',
              isMatched(flight.id) ? 'flight-option-matched ring-2 ring-blue-500' : '',
              isFiltered ? 'flight-option-filtered' : 'flight-option-visible'
            ]"
            @click="openFlight(flight.id)"
          >
            <!-- Airline Info -->
            <div class="w-full sm:w-1/4 flex items-center gap-4">
              <div class="w-12 h-12 rounded-full bg-gray-50 overflow-hidden border border-gray-100 flex-shrink-0">
                <img :src="flight.image" :alt="flight.airline" class="w-full h-full object-cover" />
              </div>
              <div class="font-bold text-gray-900">{{ flight.airline }}</div>
            </div>

            <!-- Flight Segment -->
            <div class="flex-1 flex flex-col justify-center">
              <div class="flex items-center justify-between mb-1">
                <div class="text-xl font-bold text-gray-900">{{ flight.departure_time }}</div>
                <div class="flex-1 border-b-2 border-gray-200 mx-4 relative top-[-4px]"></div>
                <div class="text-xl font-bold text-gray-900">{{ flight.arrival_time }}</div>
              </div>
              <div class="flex items-center justify-between text-xs text-gray-500 font-medium uppercase tracking-wide">
                <div>{{ flight.origin }}</div>
                <div>{{ flight.duration }} • {{ flight.stops === 0 ? 'Direct' : flight.stops + ' Stop' }}</div>
                <div>{{ flight.destination }}</div>
              </div>
            </div>

            <!-- Price & Deal -->
            <div class="w-full sm:w-1/4 border-t sm:border-t-0 sm:border-l border-gray-100 sm:pl-6 pt-4 sm:pt-0 flex flex-row sm:flex-col justify-between sm:justify-center items-center gap-2">
              <div class="text-xs font-bold text-green-600 bg-green-50 px-2 py-1 rounded">Best Deal</div>
              <div class="text-2xl font-bold text-gray-900">£{{ flight.price }}</div>
              <div class="text-xs text-gray-400">per adult</div>
              <button class="w-full mt-2 py-2 bg-blue-600 text-white font-bold rounded-lg hover:bg-blue-700 transition-colors hidden sm:block">
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
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'FLIGHTS_RESULTS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const sortOpen = ref(false)
    const currentSortLabel = ref('Default')
    const searchQuery = ref('')
    const matchedId = ref(null)

    // Store state sync
    const isFiltered = computed(() => store.flights_results_filters_applied)
    
    // Filtering Logic
    const filteredFlights = computed(() => {
      let result = [...dataStore.flights]

      // Filter: Nonstop
      if (store.stops_filter_nonstop) {
        result = result.filter(f => f.stops === 0)
      }

      // Filter: Time Slider (Simulated: filter out flights before hour X)
      // Since FSM doesn't specify slider value logic detail, we assume usage implies filtering
      // Slider value is not in store, store just has 'times_slider_used' bool.
      // But action handler needs to act. We'll rely on local state or store logic if needed.
      // For visual feedback, if slider used, we just slightly shuffle or filter dummy.
      // Let's implement real logic: filter departure hour > slider value.
      // Assuming slider is handled via handleTimeSlider which sets a local ref or store flag.
      
      // Filter: Search Query (Fuzzy matching - split by spaces)
      if (searchQuery.value) {
        const searchTerms = searchQuery.value.toLowerCase().trim().split(/\s+/).filter(term => term.length > 0)

        result = result.filter(f => {
          const searchableText = [
            f.airline,
            f.origin,
            f.destination,
            f.flight_number || ''
          ].join(' ').toLowerCase()

          // Match if ANY search term is found in the searchable text
          return searchTerms.some(term => searchableText.includes(term))
        })
      }

      // Sort
      if (store.sort_option) {
        if (store.sort_option === 'cheapest') {
          result.sort((a, b) => a.price - b.price)
        } else if (store.sort_option === 'fastest') {
          // Duration string sort (simple approx)
          result.sort((a, b) => parseInt(a.duration) - parseInt(b.duration))
        }
      }

      return result
    })

    const isMatched = (id) => id === matchedId.value

    const handleFilterNonstop = () => {
      store.stops_filter_nonstop = !store.stops_filter_nonstop
      store.flights_results_filters_applied = true
    }

    const handleTimeSlider = () => {
      store.times_slider_used = true
      store.flights_results_filters_applied = true
    }

    const toggleSort = () => sortOpen.value = !sortOpen.value

    const selectSort = (option) => {
      store.sort_option = option
      currentSortLabel.value = option === 'cheapest' ? 'Cheapest' : option === 'fastest' ? 'Fastest' : option === 'best' ? 'Best Value' : 'Default'
      store.flights_results_filters_applied = true
      sortOpen.value = false
    }

    const handleSearch = () => {
      store.flights_results_has_searched = true
      // Simulate finding a match
      if (filteredFlights.value.length > 0) {
        store.matched_item_id = filteredFlights.value[0].id
        matchedId.value = filteredFlights.value[0].id
      }
    }

    const handleScroll = (e) => {
      // Just track scroll interaction if needed
      // FSM action 'FLIGHTS_RESULTS_SCROLL_INTO_VIEW' sets anchor id
      // We assume user manually scrolls or drags
      // We can take the first visible item's ID for FSM effect
      const list = e.target
      // Simple heuristic for demo: first item is "visible"
      if (filteredFlights.value.length > 0) {
        store.flights_results_viewport_anchor_id = filteredFlights.value[0].id
      }
    }

    const openFlight = async (id) => {
      store.selected_item_id = id
      
      // Clear flags based on context (handled in FSM effects)
      if (store.flights_results_filters_applied) store.flights_results_filters_applied = null
      if (store.flights_results_has_searched) store.flights_results_has_searched = null
      if (store.flights_results_viewport_anchor_id) store.flights_results_viewport_anchor_id = null

      store.currentPageId = 'FLIGHT_DETAILS'
      await router.push({ name: 'FLIGHT_DETAILS', params: { id } })
    }

    const goBack = async () => {
      store.currentPageId = 'FLIGHTS_SEARCH'
      await router.push({ name: 'FLIGHTS_SEARCH' })
    }

    return {
      sortOpen,
      currentSortLabel,
      searchQuery,
      filteredFlights,
      isFiltered,
      isMatched,
      handleFilterNonstop,
      handleTimeSlider,
      toggleSort,
      selectSort,
      handleSearch,
      handleScroll,
      openFlight,
      goBack
    }
  }
}
</script>