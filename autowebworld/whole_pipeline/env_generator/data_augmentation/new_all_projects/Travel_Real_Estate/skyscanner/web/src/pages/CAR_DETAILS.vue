<template>
  <div class="min-h-screen bg-slate-50 font-sans pb-12">
    <header class="bg-white shadow-sm sticky top-0 z-30">
      <div class="max-w-4xl mx-auto px-6 h-16 flex items-center justify-between">
        <div id="back-cars-results" @click="goBack" class="flex items-center gap-2 cursor-pointer text-[#002D5C] hover:text-blue-600 transition-colors">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"/></svg>
          <span class="font-medium">Back to Results</span>
        </div>
        <div class="font-bold text-[#002D5C]">Car Details</div>
        <div class="w-24"></div>
      </div>
    </header>

    <main class="max-w-4xl mx-auto px-6 py-8 space-y-8" v-if="car">
      <div class="bg-white rounded-2xl shadow-sm border border-gray-100 p-8">
        <div class="flex flex-col md:flex-row gap-8 items-center">
           <div class="w-full md:w-1/2">
              <img :src="car.image" class="w-full object-contain" />
           </div>
           <div class="w-full md:w-1/2 space-y-4">
              <div>
                <h1 class="text-3xl font-bold text-gray-900">{{ car.model }}</h1>
                <p class="text-gray-500 font-medium">{{ car.type }}</p>
              </div>
              
              <div class="grid grid-cols-2 gap-4 text-sm text-gray-600">
                 <div class="flex items-center gap-2">
                   <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"/></svg>
                   {{ car.transmission }}
                 </div>
                 <div class="flex items-center gap-2">
                   <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"/></svg>
                   {{ car.seats }} Seats
                 </div>
              </div>

              <div class="pt-6 border-t border-gray-100 flex items-end justify-between">
                 <div>
                   <p class="text-xs font-bold text-gray-400 uppercase tracking-wide">Total Price</p>
                   <p class="text-3xl font-bold text-[#0770E3]">£{{ car.price }}</p>
                 </div>
              </div>
           </div>
        </div>
      </div>

      <div class="bg-white rounded-2xl shadow-sm border border-gray-100 p-8 space-y-6">
        <h2 class="text-xl font-bold text-gray-900">Driver Details</h2>
        <input 
          id="driver-name-input"
          type="text" 
          @input="handleDriverName"
          class="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none transition-all"
          placeholder="Driver's Full Name"
        />

        <button 
          id="car-validate-button"
          @click="validateForm"
          class="w-full py-4 bg-gray-100 hover:bg-gray-200 text-gray-800 font-bold rounded-xl transition-colors"
        >
          Verify Details
        </button>

        <button 
          v-if="isValid"
          id="confirm-car-booking"
          @click="confirmBooking"
          class="w-full py-5 bg-[#00A698] hover:bg-[#008f82] text-white text-xl font-bold rounded-2xl shadow-xl shadow-teal-500/20 transition-all transform hover:-translate-y-1 flex items-center justify-center gap-2"
        >
          Confirm Booking
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>
        </button>
      </div>

    </main>
  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'CAR_DETAILS',
  setup() {
    const router = useRouter()
    const route = useRoute()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const car = ref(null)
    const isValid = computed(() => store.car_form_valid)

    onMounted(() => {
      const id = route.params.id || store.car_selected_id
      car.value = dataStore.cars.find(c => c.id === id)
    })

    const handleDriverName = () => store.driver_name_entered = true

    const validateForm = () => {
      if (store.driver_name_entered) {
        store.car_form_valid = true
      }
    }

    const confirmBooking = async () => {
      if (store.car_form_valid) {
        store.currentPageId = 'BOOKING_COMPLETE_CAR'
        await router.push({ name: 'BOOKING_COMPLETE_CAR' })
      }
    }

    const goBack = async () => {
      store.currentPageId = 'CARS_RESULTS'
      await router.push({ name: 'CARS_RESULTS' })
    }

    return {
      car,
      isValid,
      handleDriverName,
      validateForm,
      confirmBooking,
      goBack
    }
  }
}
</script>