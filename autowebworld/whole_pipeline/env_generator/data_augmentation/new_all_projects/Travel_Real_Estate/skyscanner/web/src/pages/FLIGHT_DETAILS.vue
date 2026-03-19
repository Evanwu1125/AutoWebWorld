<template>
  <div class="min-h-screen bg-slate-50 font-sans pb-12">
    <header class="bg-white shadow-sm sticky top-0 z-30">
      <div class="max-w-4xl mx-auto px-6 h-16 flex items-center justify-between">
        <div id="back-results" @click="goBack" class="flex items-center gap-2 cursor-pointer text-[#002D5C] hover:text-blue-600 transition-colors">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"/></svg>
          <span class="font-medium">Back to Results</span>
        </div>
        <div class="font-bold text-[#002D5C]">Flight Details</div>
        <div class="w-24"></div>
      </div>
    </header>

    <main class="max-w-4xl mx-auto px-6 py-8 space-y-8" v-if="flight">
      <!-- Flight Summary Card -->
      <div class="bg-white rounded-2xl shadow-sm border border-gray-100 p-8">
        <div class="flex flex-col md:flex-row gap-8 items-start md:items-center justify-between">
           <div class="flex items-center gap-6">
             <img :src="flight.image" class="w-20 h-20 rounded-xl object-cover shadow-sm" />
             <div>
               <h1 class="text-2xl font-bold text-gray-900">{{ flight.airline }}</h1>
               <div class="flex items-center gap-2 text-gray-500 mt-1">
                 <span class="font-medium">{{ flight.origin }}</span>
                 <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 8l4 4m0 0l-4 4m4-4H3"/></svg>
                 <span class="font-medium">{{ flight.destination }}</span>
               </div>
             </div>
           </div>
           <div class="text-right">
             <div class="text-3xl font-bold text-[#0770E3]">£{{ flight.price }}</div>
             <div class="text-sm text-gray-400">Total for 1 adult</div>
           </div>
        </div>

        <div class="mt-8 pt-8 border-t border-gray-100 grid grid-cols-1 md:grid-cols-3 gap-6">
          <div class="bg-gray-50 p-4 rounded-xl">
             <div class="text-xs font-bold text-gray-400 uppercase tracking-wide mb-1">Departure</div>
             <div class="text-xl font-bold text-gray-900">{{ flight.departure_time }}</div>
             <div class="text-sm text-gray-500">{{ flight.origin }}</div>
          </div>
          <div class="flex flex-col items-center justify-center">
            <div class="text-sm font-medium text-gray-500 mb-1">{{ flight.duration }}</div>
            <div class="w-full h-1 bg-gray-200 rounded-full relative">
               <div class="absolute inset-0 bg-blue-200 rounded-full w-2/3"></div>
            </div>
            <div class="text-xs font-bold text-gray-400 uppercase tracking-wide mt-2">{{ flight.stops === 0 ? 'Direct' : flight.stops + ' Stop' }}</div>
          </div>
          <div class="bg-gray-50 p-4 rounded-xl text-right">
             <div class="text-xs font-bold text-gray-400 uppercase tracking-wide mb-1">Arrival</div>
             <div class="text-xl font-bold text-gray-900">{{ flight.arrival_time }}</div>
             <div class="text-sm text-gray-500">{{ flight.destination }}</div>
          </div>
        </div>
      </div>

      <!-- Add-ons Selection -->
      <div class="bg-white rounded-2xl shadow-sm border border-gray-100 p-8 space-y-8">
        <h2 class="text-xl font-bold text-gray-900">Customize Your Trip</h2>
        
        <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
          <!-- Baggage -->
          <div class="space-y-4">
            <label class="block text-sm font-bold text-gray-700">Baggage Allowance</label>
            <div class="relative group">
              <div id="baggage-dropdown" @click="toggleBaggage" class="flex items-center justify-between cursor-pointer bg-white border border-gray-200 hover:border-blue-500 px-4 py-3 rounded-xl transition-all">
                <span class="font-medium text-gray-900">{{ baggageLabel }}</span>
                <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
              </div>
              <div v-if="baggageOpen" class="absolute top-full left-0 w-full mt-2 bg-white rounded-xl shadow-xl border border-gray-100 z-50 overflow-hidden">
                <div id="baggage-none" @click="selectBaggage('no_bag')" class="px-4 py-3 hover:bg-blue-50 cursor-pointer border-b border-gray-50">
                   <div class="font-bold text-gray-900">Carry-on Only</div>
                   <div class="text-xs text-gray-500">Included</div>
                </div>
                <div id="baggage-standard" @click="selectBaggage('standard')" class="px-4 py-3 hover:bg-blue-50 cursor-pointer border-b border-gray-50">
                   <div class="font-bold text-gray-900">Standard Bag</div>
                   <div class="text-xs text-gray-500">+ £40</div>
                </div>
                <div id="baggage-extra" @click="selectBaggage('extra')" class="px-4 py-3 hover:bg-blue-50 cursor-pointer">
                   <div class="font-bold text-gray-900">Extra Heavy Bag</div>
                   <div class="text-xs text-gray-500">+ £65</div>
                </div>
              </div>
            </div>
          </div>

          <!-- Seats -->
          <div class="space-y-4">
            <label class="block text-sm font-bold text-gray-700">Seat Selection</label>
            <div class="relative group">
              <div id="seat-dropdown" @click="toggleSeat" class="flex items-center justify-between cursor-pointer bg-white border border-gray-200 hover:border-blue-500 px-4 py-3 rounded-xl transition-all">
                <span class="font-medium text-gray-900">{{ seatLabel }}</span>
                <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
              </div>
              <div v-if="seatOpen" class="absolute top-full left-0 w-full mt-2 bg-white rounded-xl shadow-xl border border-gray-100 z-50 overflow-hidden">
                <div id="seat-standard" @click="selectSeat('standard')" class="px-4 py-3 hover:bg-blue-50 cursor-pointer border-b border-gray-50">
                   <div class="font-bold text-gray-900">Standard Seat</div>
                   <div class="text-xs text-gray-500">Free</div>
                </div>
                <div id="seat-extra" @click="selectSeat('extra_legroom')" class="px-4 py-3 hover:bg-blue-50 cursor-pointer">
                   <div class="font-bold text-gray-900">Extra Legroom</div>
                   <div class="text-xs text-gray-500">+ £25</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <button 
          id="extras-validate-button"
          @click="validateExtras"
          class="w-full py-4 bg-gray-100 hover:bg-gray-200 text-gray-800 font-bold rounded-xl transition-colors"
        >
          Confirm Selections
        </button>
      </div>

      <!-- Price Alert CTA -->
      <div class="bg-blue-50 rounded-2xl p-6 border border-blue-100 flex flex-col sm:flex-row items-center justify-between gap-6">
        <div>
          <h3 class="text-lg font-bold text-[#002D5C] mb-1">Not ready to book?</h3>
          <p class="text-blue-600/80 text-sm">Track prices for this flight and get notified when they change.</p>
        </div>
        <button 
          id="create-price-alert-button"
          @click="createAlert"
          class="shrink-0 px-6 py-3 bg-white text-blue-600 font-bold rounded-xl shadow-sm border border-blue-200 hover:bg-blue-50 transition-colors"
        >
          Create Price Alert
        </button>
      </div>

      <!-- Main CTA -->
      <button 
        v-if="isValid"
        id="continue-to-booking"
        @click="goToBooking"
        class="w-full py-5 bg-[#00A698] hover:bg-[#008f82] text-white text-xl font-bold rounded-2xl shadow-xl shadow-teal-500/20 transition-all transform hover:-translate-y-1 flex items-center justify-center gap-2"
      >
        Continue to Booking
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"/></svg>
      </button>

    </main>
  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'FLIGHT_DETAILS',
  setup() {
    const router = useRouter()
    const route = useRoute()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const flight = ref(null)
    const baggageOpen = ref(false)
    const seatOpen = ref(false)
    const baggageLabel = ref('Select Baggage')
    const seatLabel = ref('Select Seat')

    const isValid = computed(() => store.extras_form_valid)

    onMounted(() => {
      // Get flight from ID
      const id = route.params.id || store.selected_item_id
      flight.value = dataStore.flights.find(f => f.id === id)
    })

    const toggleBaggage = () => baggageOpen.value = !baggageOpen.value
    const toggleSeat = () => seatOpen.value = !seatOpen.value

    const selectBaggage = (option) => {
      store.selected_baggage_option = option
      baggageLabel.value = option === 'no_bag' ? 'Carry-on Only' : option === 'standard' ? 'Standard Bag' : 'Extra Heavy Bag'
      baggageOpen.value = false
    }

    const selectSeat = (option) => {
      store.selected_seat_option = option
      seatLabel.value = option === 'standard' ? 'Standard Seat' : 'Extra Legroom'
      seatOpen.value = false
    }

    const validateExtras = () => {
      if (store.selected_baggage_option && store.selected_seat_option) {
        store.extras_form_valid = true
      }
    }

    const goToBooking = async () => {
      store.currentPageId = 'BOOKING_FORM_DIRECT'
      await router.push({ name: 'BOOKING_FORM_DIRECT' })
    }

    const createAlert = async () => {
      store.currentPageId = 'PRICE_ALERT_FORM'
      await router.push({ name: 'PRICE_ALERT_FORM' })
    }

    const goBack = async () => {
      store.currentPageId = 'FLIGHTS_RESULTS'
      await router.push({ name: 'FLIGHTS_RESULTS' })
    }

    return {
      flight,
      baggageOpen,
      seatOpen,
      baggageLabel,
      seatLabel,
      isValid,
      toggleBaggage,
      toggleSeat,
      selectBaggage,
      selectSeat,
      validateExtras,
      goToBooking,
      createAlert,
      goBack
    }
  }
}
</script>