<template>
  <div class="min-h-screen bg-slate-50 font-sans">
    <header class="bg-[#002D5C] text-white py-4 px-6 shadow-md sticky top-0 z-30">
      <div class="max-w-7xl mx-auto flex items-center justify-between">
        <div id="back-cars-search" @click="goBack" class="flex items-center gap-2 cursor-pointer hover:text-blue-200 transition-colors">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"/></svg>
          <span class="font-medium">Modify Search</span>
        </div>
        <div class="font-bold text-xl">Car Hire Results</div>
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
                id="filter-automatic-checkbox"
                type="checkbox" 
                @change="handleFilterAutomatic"
                class="peer h-5 w-5 cursor-pointer appearance-none rounded border border-gray-300 shadow-sm transition-all checked:border-blue-600 checked:bg-blue-600 hover:border-blue-400 focus:ring-2 focus:ring-blue-200" 
              />
              <svg class="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-3.5 h-3.5 pointer-events-none opacity-0 peer-checked:opacity-100 text-white" viewBox="0 0 14 14" fill="none">
                <path d="M3 8L6 11L11 3.5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
              </svg>
            </div>
            <span class="text-gray-700 group-hover:text-blue-600 transition-colors">Automatic Transmission</span>
          </label>
        </div>
      </aside>

      <!-- Results -->
      <div class="flex-1 space-y-6">
        <div id="cars-results-list" @scroll="handleScroll" class="grid grid-cols-1 gap-6">
          <div 
            v-for="car in filteredCars" 
            :key="car.id"
            :class="[
              `data-id-${car.id}`,
              'bg-white rounded-2xl shadow-sm hover:shadow-xl transition-all border border-gray-100 p-6 flex flex-col sm:flex-row gap-6 cursor-pointer group',
              isFiltered ? 'car-row-filtered' : 'car-row-visible'
            ]"
            @click="openCar(car.id)"
          >
            <div class="w-full sm:w-1/3 flex items-center justify-center bg-gray-50 rounded-xl p-4">
              <img :src="car.image" :alt="car.model" class="max-w-full max-h-32 object-contain group-hover:scale-110 transition-transform duration-500" />
            </div>
            
            <div class="flex-1 flex flex-col justify-between">
              <div>
                <h3 class="text-xl font-bold text-gray-900 mb-1 group-hover:text-blue-600 transition-colors">{{ car.model }}</h3>
                <p class="text-sm text-gray-500">{{ car.type }} • {{ car.transmission }}</p>
                <div class="flex items-center gap-4 mt-4 text-sm text-gray-600">
                   <div class="flex items-center gap-1">
                      <svg class="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"/></svg>
                      {{ car.seats }} Seats
                   </div>
                   <div class="flex items-center gap-1">
                      <svg class="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"/></svg>
                      Air Conditioning
                   </div>
                </div>
              </div>

              <div class="flex items-end justify-between mt-4">
                 <div class="text-green-600 text-sm font-bold flex items-center gap-1">
                   <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>
                   Free Cancellation
                 </div>
                 <div class="text-right">
                   <div class="text-2xl font-bold text-gray-900">£{{ car.price }}</div>
                   <div class="text-xs text-gray-400">per day</div>
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
  name: 'CARS_RESULTS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const isFiltered = computed(() => store.cars_results_filters_applied)
    const automaticChecked = ref(false)

    const filteredCars = computed(() => {
      let result = [...dataStore.cars]
      if (automaticChecked.value) {
        result = result.filter(c => c.transmission === 'Automatic')
      }
      return result
    })

    const handleFilterAutomatic = () => {
      automaticChecked.value = !automaticChecked.value
      store.cars_results_filters_applied = true
    }

    const handleScroll = (e) => {
      if (filteredCars.value.length > 0) {
        store.cars_results_viewport_anchor_id = filteredCars.value[0].id
      }
    }

    const openCar = async (id) => {
      store.car_selected_id = id
      
      // Clear flags
      if (store.cars_results_filters_applied) store.cars_results_filters_applied = null
      if (store.cars_results_viewport_anchor_id) store.cars_results_viewport_anchor_id = null

      store.currentPageId = 'CAR_DETAILS'
      await router.push({ name: 'CAR_DETAILS', params: { id } })
    }

    const goBack = async () => {
      store.currentPageId = 'CARS_SEARCH'
      await router.push({ name: 'CARS_SEARCH' })
    }

    return {
      filteredCars,
      isFiltered,
      handleFilterAutomatic,
      handleScroll,
      openCar,
      goBack
    }
  }
}
</script>