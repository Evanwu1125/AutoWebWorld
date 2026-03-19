<template>
  <div class="min-h-screen bg-slate-50 font-sans">
    <header class="bg-[#002D5C] text-white py-4 px-6 shadow-md">
      <div class="max-w-7xl mx-auto flex items-center justify-between">
        <div id="back-home-from-cars" @click="goHome" class="flex items-center gap-2 cursor-pointer hover:text-blue-200 transition-colors">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"/></svg>
          <span class="font-medium">Back to Home</span>
        </div>
        <div class="font-bold text-xl">Car Hire</div>
        <div class="w-24"></div>
      </div>
    </header>

    <main class="max-w-4xl mx-auto px-6 py-12">
      <div class="bg-white rounded-2xl shadow-xl overflow-hidden p-8 space-y-8">
        <h1 class="text-3xl font-bold text-gray-900">Hit the road</h1>
        
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 relative">
          <!-- Pickup -->
          <div class="relative">
            <label class="block text-xs font-bold text-gray-500 uppercase tracking-wide mb-1 ml-1">Pickup Location</label>
            <div class="relative">
              <div class="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                 <svg class="h-5 w-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"/></svg>
              </div>
              <input 
                id="car-pickup-input"
                type="text" 
                @input="handlePickupInput"
                class="block w-full pl-11 pr-4 py-4 bg-gray-50 border-none rounded-xl text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all font-semibold text-lg"
                placeholder="Airport or City"
              />
            </div>
          </div>

          <!-- Dates -->
          <div>
            <label class="block text-xs font-bold text-gray-500 uppercase tracking-wide mb-1 ml-1">Pickup - Dropoff</label>
              <DateTimePicker id="date-picker3" @change="handleDateChange" />
          </div>
        </div>

        <div class="flex gap-4 pt-4 border-t border-gray-100">
           <button 
              id="car-search-validate-button"
              @click="validateSearch"
              class="flex-1 py-4 bg-gray-100 hover:bg-gray-200 text-gray-800 font-bold rounded-xl transition-colors"
            >
              Check Availability
            </button>
            <button 
              v-if="isReady"
              id="search-cars-button"
              @click="submitSearch"
              class="flex-[2] py-4 bg-[#0770E3] hover:bg-[#0660C3] text-white font-bold rounded-xl shadow-lg shadow-blue-600/20 transition-all transform hover:-translate-y-0.5 flex items-center justify-center gap-2"
            >
              Search Cars
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
            </button>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import DateTimePicker from '../components/widgets/DateTimePicker.vue'

export default {
  name: 'CARS_SEARCH',
  components: {
    DateTimePicker
  },
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const isReady = computed(() => store.car_search_ready)

    const handlePickupInput = () => store.car_pickup_entered = true
    const handleDateChange = () => store.car_dates_selected = true

    const validateSearch = () => {
      if (store.car_pickup_entered && store.car_dates_selected) {
        store.car_search_ready = true
      }
    }

    const submitSearch = async () => {
      if (store.car_search_ready) {
        store.currentPageId = 'CARS_RESULTS'
        await router.push({ name: 'CARS_RESULTS' })
      }
    }

    const goHome = async () => {
      store.currentPageId = 'HOME'
      await router.push({ name: 'HOME' })
    }

    return {
      isReady,
      handlePickupInput,
      handleDateChange,
      validateSearch,
      submitSearch,
      goHome
    }
  }
}
</script>