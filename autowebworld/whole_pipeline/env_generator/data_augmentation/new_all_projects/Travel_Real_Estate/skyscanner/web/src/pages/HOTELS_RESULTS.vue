<template>
  <div class="min-h-screen bg-slate-50 font-sans">
    <header class="bg-[#002D5C] text-white py-4 px-6 shadow-md sticky top-0 z-30">
      <div class="max-w-7xl mx-auto flex items-center justify-between">
        <div id="back-hotels-search" @click="goBack" class="flex items-center gap-2 cursor-pointer hover:text-blue-200 transition-colors">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"/></svg>
          <span class="font-medium">Modify Search</span>
        </div>
        <div class="font-bold text-xl">Hotels in London</div>
        <div class="w-24"></div>
      </div>
    </header>

    <div class="max-w-7xl mx-auto px-4 md:px-6 py-8 flex flex-col md:flex-row gap-8">
      <!-- Sidebar -->
      <aside class="w-full md:w-72 shrink-0 space-y-6">
        <div class="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
          <h3 class="font-bold text-gray-900 mb-4 border-b border-gray-100 pb-2">Filter</h3>
           <label class="flex items-center gap-3 cursor-pointer group">
            <div class="relative flex items-center">
              <input 
                id="filter-free-cancellation-checkbox"
                type="checkbox" 
                @change="handleFilterFreeCancel"
                class="peer h-5 w-5 cursor-pointer appearance-none rounded border border-gray-300 shadow-sm transition-all checked:border-blue-600 checked:bg-blue-600 hover:border-blue-400 focus:ring-2 focus:ring-blue-200" 
              />
              <svg class="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-3.5 h-3.5 pointer-events-none opacity-0 peer-checked:opacity-100 text-white" viewBox="0 0 14 14" fill="none">
                <path d="M3 8L6 11L11 3.5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
              </svg>
            </div>
            <span class="text-gray-700 group-hover:text-blue-600 transition-colors">Free Cancellation</span>
          </label>
        </div>
      </aside>

      <!-- Results -->
      <div class="flex-1 space-y-6">
        <div class="flex justify-end">
           <div class="relative">
            <div id="hotels-sort-dropdown" @click="toggleSort" class="flex items-center gap-2 cursor-pointer text-gray-700 font-medium hover:bg-gray-50 px-4 py-2 rounded-lg border border-gray-200 transition-colors bg-white">
              <span class="text-sm">Sort by: <span class="text-blue-600">{{ currentSortLabel }}</span></span>
              <svg class="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
            </div>
            <div v-if="sortOpen" class="absolute top-full right-0 mt-2 w-48 bg-white rounded-xl shadow-xl border border-gray-100 z-50 overflow-hidden">
               <div id="hotels-sort-recommended" @click="selectSort('recommended')" class="px-4 py-3 hover:bg-blue-50 cursor-pointer text-gray-700 text-sm">Recommended</div>
               <div id="hotels-sort-price-low" @click="selectSort('price_low')" class="px-4 py-3 hover:bg-blue-50 cursor-pointer text-gray-700 text-sm">Price (Low to High)</div>
               <div id="hotels-sort-price-high" @click="selectSort('price_high')" class="px-4 py-3 hover:bg-blue-50 cursor-pointer text-gray-700 text-sm">Price (High to Low)</div>
            </div>
          </div>
        </div>

        <div id="hotels-results-list" @scroll="handleScroll" class="grid grid-cols-1 gap-6">
          <div 
            v-for="hotel in filteredHotels" 
            :key="hotel.id"
            :class="[
              `data-id-${hotel.id}`,
              'bg-white rounded-2xl shadow-sm hover:shadow-xl transition-all border border-gray-100 overflow-hidden cursor-pointer flex flex-col sm:flex-row h-auto sm:h-56 group',
              isFiltered ? 'hotel-row-filtered' : 'hotel-row-visible'
            ]"
            @click="openHotel(hotel.id)"
          >
            <div class="w-full sm:w-1/3 relative overflow-hidden">
              <img :src="hotel.image" :alt="hotel.name" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" />
              <div v-if="hotel.free_cancellation" class="absolute top-2 left-2 bg-green-500 text-white text-xs font-bold px-2 py-1 rounded shadow-sm">Free Cancellation</div>
            </div>
            
            <div class="p-6 flex-1 flex flex-col justify-between">
              <div>
                <div class="flex justify-between items-start">
                  <div>
                    <h3 class="text-xl font-bold text-gray-900 mb-1 group-hover:text-blue-600 transition-colors">{{ hotel.name }}</h3>
                    <p class="text-sm text-gray-500 flex items-center gap-1">
                      <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"/></svg>
                      {{ hotel.location }}
                    </p>
                  </div>
                  <div class="flex text-yellow-400">
                     <span v-for="i in 5" :key="i" class="text-sm">{{ i <= hotel.stars ? '★' : '☆' }}</span>
                  </div>
                </div>
              </div>

              <div class="flex items-end justify-between mt-4">
                 <div>
                   <div class="inline-flex items-center gap-1 bg-blue-100 text-blue-800 text-xs font-bold px-2 py-1 rounded">
                     {{ hotel.rating }}/5 Very Good
                   </div>
                 </div>
                 <div class="text-right">
                   <div class="text-xs text-gray-400">from</div>
                   <div class="text-2xl font-bold text-gray-900">£{{ hotel.price }}</div>
                   <div class="text-xs text-gray-400">per night</div>
                 </div>
              </div>
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
  name: 'HOTELS_RESULTS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const sortOpen = ref(false)
    const currentSortLabel = ref('Default')

    const isFiltered = computed(() => store.hotels_results_filters_applied)

    const filteredHotels = computed(() => {
      let result = [...dataStore.hotels]
      // Since FSM check `hotels_results_filters_applied` precondition for `HOTELS_RESULTS_OPEN_FILTERED_HOTEL`,
      // we need to actually filter something to show "filtered" items if the flag is true.
      // But user sets the flag via action.
      // If flag is true (via free cancellation checkbox), filter by free cancellation.
      // FSM doesn't store checkbox state in signature, just `hotels_results_filters_applied`.
      // We'll rely on a local ref for checkbox state, or assume the flag implies it.
      // Wait, signature has `hotels_results_filters_applied` but no `free_cancellation_checked`.
      // The action `HOTELS_RESULTS_FILTER_FREE_CANCELLATION` sets `hotels_results_filters_applied` to true.
      // So if that flag is true, we should filter. But what if user unchecks?
      // The FSM is simple: action -> set flag. It doesn't model "uncheck".
      // We will implement toggle locally and set flag on change.
      // However, for FSM compliance, we need to respect the flag logic.
      
      // Let's rely on local state for actual filtering logic, but ensure flag is set as per FSM action.
      // We can check if `store.hotels_results_filters_applied` is true, then apply "some" filter.
      // But better: use local state `isFreeCancel` and sync flag.
      // Wait, simpler: if user clicks checkbox, trigger action.
      // The filter logic below uses local state `freeCancelChecked`.
      if (freeCancelChecked.value) {
        result = result.filter(h => h.free_cancellation)
      }

      if (store.hotels_sort_option) {
        if (store.hotels_sort_option === 'price_low') {
          result.sort((a, b) => a.price - b.price)
        } else if (store.hotels_sort_option === 'price_high') {
          result.sort((a, b) => b.price - a.price)
        } else if (store.hotels_sort_option === 'recommended') {
          // Keep default order for recommended
        }
      }

      return result
    })

    const freeCancelChecked = ref(false)
    const handleFilterFreeCancel = () => {
      freeCancelChecked.value = !freeCancelChecked.value
      store.hotels_results_filters_applied = true
    }

    const toggleSort = () => sortOpen.value = !sortOpen.value

    const selectSort = (option) => {
      store.hotels_sort_option = option
      currentSortLabel.value = option === 'recommended' ? 'Recommended' : option === 'price_low' ? 'Price (Low to High)' : option === 'price_high' ? 'Price (High to Low)' : 'Default'
      store.hotels_results_filters_applied = true
      sortOpen.value = false
    }

    const handleScroll = (e) => {
      if (filteredHotels.value.length > 0) {
        store.hotels_results_viewport_anchor_id = filteredHotels.value[0].id
      }
    }

    const openHotel = async (id) => {
      store.hotel_selected_id = id
      
      // Clear flags
      if (store.hotels_results_filters_applied) store.hotels_results_filters_applied = null
      if (store.hotels_results_viewport_anchor_id) store.hotels_results_viewport_anchor_id = null

      store.currentPageId = 'HOTEL_DETAILS'
      await router.push({ name: 'HOTEL_DETAILS', params: { id } })
    }

    const goBack = async () => {
      store.currentPageId = 'HOTELS_SEARCH'
      await router.push({ name: 'HOTELS_SEARCH' })
    }

    return {
      sortOpen,
      currentSortLabel,
      filteredHotels,
      isFiltered,
      handleFilterFreeCancel,
      toggleSort,
      selectSort,
      handleScroll,
      openHotel,
      goBack
    }
  }
}
</script>