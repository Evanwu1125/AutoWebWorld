<template>
  <div class="min-h-screen bg-slate-50 font-sans pb-12">
    <header class="bg-white shadow-sm sticky top-0 z-30">
      <div class="max-w-4xl mx-auto px-6 h-16 flex items-center justify-between">
        <div id="back-hotels-results" @click="goBack" class="flex items-center gap-2 cursor-pointer text-[#002D5C] hover:text-blue-600 transition-colors">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"/></svg>
          <span class="font-medium">Back to Results</span>
        </div>
        <div class="font-bold text-[#002D5C]">Hotel Details</div>
        <div class="w-24"></div>
      </div>
    </header>

    <main class="max-w-4xl mx-auto px-6 py-8 space-y-8" v-if="hotel">
      <div class="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
        <div class="h-64 sm:h-80 relative">
           <img :src="hotel.image" class="w-full h-full object-cover" />
           <div class="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent"></div>
           <div class="absolute bottom-6 left-6 right-6 text-white flex flex-col sm:flex-row justify-between items-end gap-4">
             <div>
               <h1 class="text-3xl md:text-4xl font-bold mb-2">{{ hotel.name }}</h1>
               <div class="flex items-center gap-2 text-white/90">
                 <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"/></svg>
                 <span class="font-medium text-lg">{{ hotel.location }}</span>
               </div>
             </div>
             <div class="text-right">
                <div class="text-3xl font-bold">£{{ hotel.price }}</div>
                <div class="text-sm opacity-80">per night</div>
             </div>
           </div>
        </div>
        
        <div class="p-8 border-t border-gray-100 grid grid-cols-1 md:grid-cols-2 gap-8">
          <div class="space-y-6">
            <h2 class="text-xl font-bold text-gray-900">Select Room</h2>
            <div class="relative group">
              <div id="room-type-dropdown" @click="toggleRoom" class="flex items-center justify-between cursor-pointer bg-white border border-gray-200 hover:border-blue-500 px-4 py-3 rounded-xl transition-all">
                <span class="font-medium text-gray-900">{{ roomLabel }}</span>
                <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
              </div>
              <div v-if="roomOpen" class="absolute top-full left-0 w-full mt-2 bg-white rounded-xl shadow-xl border border-gray-100 z-50 overflow-hidden">
                <div id="room-type-standard" @click="selectRoom('standard')" class="px-4 py-3 hover:bg-blue-50 cursor-pointer border-b border-gray-50">
                   <div class="font-bold text-gray-900">Standard Room</div>
                   <div class="text-xs text-gray-500">1 Queen Bed</div>
                </div>
                <div id="room-type-deluxe" @click="selectRoom('deluxe')" class="px-4 py-3 hover:bg-blue-50 cursor-pointer">
                   <div class="font-bold text-gray-900">Deluxe Room</div>
                   <div class="text-xs text-gray-500">1 King Bed + City View (+£50)</div>
                </div>
              </div>
            </div>
          </div>

          <div class="space-y-6">
            <h2 class="text-xl font-bold text-gray-900">Guest Info</h2>
             <input 
              id="guest-name-input"
              type="text" 
              @input="handleGuestName"
              class="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none transition-all"
              placeholder="Guest Full Name"
            />
          </div>
        </div>
      </div>

      <button 
        id="hotel-validate-button"
        @click="validateForm"
        class="w-full py-4 bg-gray-100 hover:bg-gray-200 text-gray-800 font-bold rounded-xl transition-colors"
      >
        Review Booking
      </button>

      <button 
        v-if="isValid"
        id="confirm-hotel-booking"
        @click="confirmBooking"
        class="w-full py-5 bg-[#00A698] hover:bg-[#008f82] text-white text-xl font-bold rounded-2xl shadow-xl shadow-teal-500/20 transition-all transform hover:-translate-y-1 flex items-center justify-center gap-2"
      >
        Confirm Booking
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>
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
  name: 'HOTEL_DETAILS',
  setup() {
    const router = useRouter()
    const route = useRoute()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const hotel = ref(null)
    const roomOpen = ref(false)
    const roomLabel = ref('Choose a Room')

    const isValid = computed(() => store.hotel_form_valid)

    onMounted(() => {
      const id = route.params.id || store.hotel_selected_id
      hotel.value = dataStore.hotels.find(h => h.id === id)
    })

    const toggleRoom = () => roomOpen.value = !roomOpen.value

    const selectRoom = (type) => {
      store.room_type_selected = true
      roomLabel.value = type === 'standard' ? 'Standard Room' : 'Deluxe Room'
      roomOpen.value = false
    }

    const handleGuestName = () => store.guest_name_entered = true

    const validateForm = () => {
      if (store.room_type_selected && store.guest_name_entered) {
        store.hotel_form_valid = true
      }
    }

    const confirmBooking = async () => {
      if (store.hotel_form_valid) {
        store.currentPageId = 'BOOKING_COMPLETE_HOTEL'
        await router.push({ name: 'BOOKING_COMPLETE_HOTEL' })
      }
    }

    const goBack = async () => {
      store.currentPageId = 'HOTELS_RESULTS'
      await router.push({ name: 'HOTELS_RESULTS' })
    }

    return {
      hotel,
      roomOpen,
      roomLabel,
      isValid,
      toggleRoom,
      selectRoom,
      handleGuestName,
      validateForm,
      confirmBooking,
      goBack
    }
  }
}
</script>