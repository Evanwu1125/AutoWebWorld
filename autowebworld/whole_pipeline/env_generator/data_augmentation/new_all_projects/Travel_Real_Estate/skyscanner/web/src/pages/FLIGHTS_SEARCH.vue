<template>
  <div class="min-h-screen bg-slate-50 font-sans">
    <PermissionModal />
    
    <header class="bg-[#002D5C] text-white py-4 px-6 shadow-md">
      <div class="max-w-7xl mx-auto flex items-center justify-between">
        <div id="back-home" @click="goHome" class="flex items-center gap-2 cursor-pointer hover:text-blue-200 transition-colors">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"/></svg>
          <span class="font-medium">Back to Home</span>
        </div>
        <div class="font-bold text-xl">Flight Search</div>
        <div class="w-24"></div> <!-- Spacer -->
      </div>
    </header>

    <main class="max-w-4xl mx-auto px-6 py-12">
      <div class="bg-white rounded-2xl shadow-xl overflow-hidden">
        <!-- Tabs -->
        <div class="flex border-b border-gray-200">
          <button class="flex-1 py-4 text-blue-600 font-bold border-b-2 border-blue-600 bg-blue-50/50">
            Return / One-way
          </button>
          <button 
            id="trip-type-multicity-tab"
            @click="goToMultiCity"
            class="flex-1 py-4 text-gray-500 font-medium hover:text-gray-700 hover:bg-gray-50 transition-colors"
          >
            Multi-city
          </button>
        </div>

        <div class="p-8 space-y-8">
          <!-- Trip Type & Cabin Row -->
          <div class="flex flex-wrap gap-4">
            <div class="relative group">
              <div id="trip-type-dropdown" @click="toggleTripType" class="flex items-center gap-2 cursor-pointer text-gray-700 font-medium hover:bg-gray-100 px-3 py-2 rounded-lg transition-colors">
                <span>{{ currentTripTypeLabel }}</span>
                <svg class="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
              </div>
              <!-- Dropdown -->
              <div v-if="tripTypeOpen" class="absolute top-full left-0 mt-2 w-48 bg-white rounded-xl shadow-xl border border-gray-100 z-50 overflow-hidden">
                <div id="trip-type-return" @click="selectTripType('return')" class="px-4 py-3 hover:bg-blue-50 cursor-pointer text-gray-700">Return</div>
                <div id="trip-type-oneway" @click="selectTripType('oneway')" class="px-4 py-3 hover:bg-blue-50 cursor-pointer text-gray-700">One-way</div>
                <div id="trip-type-multicity" @click="selectTripType('multicity')" class="px-4 py-3 hover:bg-blue-50 cursor-pointer text-gray-700">Multi-city</div>
              </div>
            </div>

            <div class="relative group">
              <div id="cabin-dropdown" @click="toggleCabin" class="flex items-center gap-2 cursor-pointer text-gray-700 font-medium hover:bg-gray-100 px-3 py-2 rounded-lg transition-colors">
                <span>{{ currentCabinLabel }}</span>
                <svg class="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
              </div>
              <div v-if="cabinOpen" class="absolute top-full left-0 mt-2 w-48 bg-white rounded-xl shadow-xl border border-gray-100 z-50 overflow-hidden">
                <div id="cabin-economy" @click="selectCabin('economy')" class="px-4 py-3 hover:bg-blue-50 cursor-pointer text-gray-700">Economy</div>
                <div id="cabin-premium" @click="selectCabin('premium')" class="px-4 py-3 hover:bg-blue-50 cursor-pointer text-gray-700">Premium Economy</div>
                <div id="cabin-business" @click="selectCabin('business')" class="px-4 py-3 hover:bg-blue-50 cursor-pointer text-gray-700">Business Class</div>
              </div>
            </div>
          </div>

          <!-- Inputs Grid -->
          <div class="grid grid-cols-1 md:grid-cols-2 gap-6 relative">
            <!-- Location Inputs -->
            <div class="space-y-6">
              <div class="relative">
                <label class="block text-xs font-bold text-gray-500 uppercase tracking-wide mb-1 ml-1">From</label>
                <div class="relative">
                  <div class="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                    <svg class="h-5 w-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"/></svg>
                  </div>
                  <input 
                    id="origin-input"
                    type="text" 
                    @input="handleOriginInput"
                    class="block w-full pl-11 pr-4 py-4 bg-gray-50 border-none rounded-xl text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all font-semibold text-lg"
                    placeholder="Country, city or airport"
                  />
                </div>
              </div>
              
              <div class="relative">
                <label class="block text-xs font-bold text-gray-500 uppercase tracking-wide mb-1 ml-1">To</label>
                <div class="relative">
                  <div class="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                    <svg class="h-5 w-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"/></svg>
                  </div>
                  <input 
                    id="destination-input"
                    type="text" 
                    @input="handleDestinationInput"
                    class="block w-full pl-11 pr-4 py-4 bg-gray-50 border-none rounded-xl text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all font-semibold text-lg"
                    placeholder="Country, city or airport"
                  />
                </div>
              </div>
            </div>

            <!-- Date Picker -->
            <div>
              <label class="block text-xs font-bold text-gray-500 uppercase tracking-wide mb-1 ml-1">Depart - Return</label>
                <DateTimePicker id="date-picker1" @change="handleDateChange" />
            </div>
          </div>

          <!-- Action Buttons -->
          <div class="flex gap-4 pt-4 border-t border-gray-100">
            <button
              id="search-validate-button"
              @click="validateSearch"
              class="flex-1 py-4 bg-gray-100 hover:bg-gray-200 text-gray-800 font-bold rounded-xl transition-colors"
            >
              Check Availability
            </button>
            <button
              id="search-flights-button"
              @click="submitSearch"
              :disabled="!isReady"
              class="flex-[2] py-4 font-bold rounded-xl shadow-lg transition-all transform flex items-center justify-center gap-2"
              :class="isReady ? 'bg-[#0770E3] hover:bg-[#0660C3] text-white shadow-blue-600/20 hover:-translate-y-0.5 cursor-pointer' : 'bg-gray-300 text-gray-500 cursor-not-allowed'"
            >
              Search Flights
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"/></svg>
            </button>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import DateTimePicker from '../components/widgets/DateTimePicker.vue'
import PermissionModal from '../components/PermissionModal.vue'

export default {
  name: 'FLIGHTS_SEARCH',
  components: {
    DateTimePicker,
    PermissionModal
  },
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const tripTypeOpen = ref(false)
    const cabinOpen = ref(false)
    const currentTripTypeLabel = ref('Return')
    const currentCabinLabel = ref('Economy')

    const isReady = computed(() => store.search_ready)

    // Set default cabin selection on mount
    onMounted(() => {
      store.cabin_selected = true
      console.log('Component mounted - cabin_selected set to true by default')
    })

    const toggleTripType = () => tripTypeOpen.value = !tripTypeOpen.value
    const toggleCabin = () => cabinOpen.value = !cabinOpen.value

    const selectTripType = (type) => {
      store.trip_type = type
      currentTripTypeLabel.value = type === 'return' ? 'Return' : type === 'oneway' ? 'One-way' : 'Multi-city'
      tripTypeOpen.value = false
      if (type === 'multicity') {
        goToMultiCity()
      }
    }

    const selectCabin = (cabin) => {
      store.cabin_selected = true
      currentCabinLabel.value = cabin.charAt(0).toUpperCase() + cabin.slice(1)
      cabinOpen.value = false
    }

    const handleOriginInput = () => {
      store.origin_entered = true
      console.log('Origin entered')
    }

    const handleDestinationInput = () => {
      store.destination_entered = true
      console.log('Destination entered')
    }

    const handleDateChange = () => {
      store.dates_selected = true
      console.log('Dates selected')
    }

    const validateSearch = () => {
      console.log('=== Validating Search ===')
      store.dates_selected = true
      console.log('origin_entered:', store.origin_entered)
      console.log('destination_entered:', store.destination_entered)
      console.log('dates_selected:', store.dates_selected)
      console.log('cabin_selected:', store.cabin_selected)

      if (store.origin_entered && store.destination_entered && store.dates_selected && store.cabin_selected) {
        console.log('✅ All conditions met - enabling search')
        store.search_ready = true
      } else {
        console.log('❌ Some conditions not met - search disabled')
        store.search_ready = false
      }

      console.log('search_ready:', store.search_ready)
    }

    const submitSearch = async () => {
      if (store.search_ready) {
        store.currentPageId = 'FLIGHTS_RESULTS'
        await router.push({ name: 'FLIGHTS_RESULTS' })
      }
    }

    const goToMultiCity = async () => {
      store.currentPageId = 'MULTI_CITY_SEARCH'
      await router.push({ name: 'MULTI_CITY_SEARCH' })
    }

    const goHome = async () => {
      store.currentPageId = 'HOME'
      await router.push({ name: 'HOME' })
    }

    return {
      tripTypeOpen,
      cabinOpen,
      currentTripTypeLabel,
      currentCabinLabel,
      isReady,
      toggleTripType,
      toggleCabin,
      selectTripType,
      selectCabin,
      handleOriginInput,
      handleDestinationInput,
      handleDateChange,
      validateSearch,
      submitSearch,
      goToMultiCity,
      goHome
    }
  }
}
</script>