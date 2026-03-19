<template>
  <div class="min-h-screen bg-slate-50 font-sans pb-12">
    <header class="bg-white shadow-sm sticky top-0 z-30">
      <div class="max-w-3xl mx-auto px-6 h-16 flex items-center justify-between">
        <div id="back-multi-intro" @click="goBack" class="flex items-center gap-2 cursor-pointer text-[#002D5C] hover:text-blue-600 transition-colors">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"/></svg>
          <span class="font-medium">Back to Itinerary</span>
        </div>
        <div class="font-bold text-[#002D5C]">Passenger Details</div>
        <div class="w-24"></div>
      </div>
    </header>

    <main class="max-w-3xl mx-auto px-6 py-8 space-y-6">
      <!-- Passenger 1 -->
      <div class="bg-white rounded-2xl shadow-sm border border-gray-100 p-8 space-y-4">
        <h2 class="text-xl font-bold text-gray-900 border-b border-gray-100 pb-4">Primary Passenger</h2>
        <input 
          id="primary-name-input"
          type="text" 
          @input="handlePrimary"
          class="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none transition-all"
          placeholder="Full Name"
        />
      </div>

      <!-- Passenger 2 -->
      <div class="bg-white rounded-2xl shadow-sm border border-gray-100 p-8 space-y-4">
        <h2 class="text-xl font-bold text-gray-900 border-b border-gray-100 pb-4">Second Passenger</h2>
        <input 
          id="second-name-input"
          type="text" 
          @input="handleSecond"
          class="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none transition-all"
          placeholder="Full Name"
        />
      </div>

      <!-- Payment -->
      <div class="bg-white rounded-2xl shadow-sm border border-gray-100 p-8 space-y-4">
        <h2 class="text-xl font-bold text-gray-900 border-b border-gray-100 pb-4">Payment</h2>
        <input 
          id="multi-card-input"
          type="text" 
          @input="handleCard"
          class="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none transition-all"
          placeholder="Card Number"
        />
      </div>

      <button 
        id="multi-booking-validate-button"
        @click="validateForm"
        class="w-full py-4 bg-gray-100 hover:bg-gray-200 text-gray-800 font-bold rounded-xl transition-colors"
      >
        Verify Details
      </button>

      <button 
        v-if="isValid"
        id="continue-multi-review"
        @click="submitForm"
        class="w-full py-5 bg-[#00A698] hover:bg-[#008f82] text-white text-xl font-bold rounded-2xl shadow-xl shadow-teal-500/20 transition-all transform hover:-translate-y-1 flex items-center justify-center gap-2"
      >
        Review Booking
      </button>
    </main>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'BOOKING_FORM_MULTI',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const isValid = computed(() => store.multi_booking_valid)

    const handlePrimary = () => store.primary_passenger_entered = true
    const handleSecond = () => store.second_passenger_entered = true
    const handleCard = () => store.multi_payment_entered = true

    const validateForm = () => {
      if (store.primary_passenger_entered && store.second_passenger_entered && store.multi_payment_entered) {
        store.multi_booking_valid = true
      }
    }

    const submitForm = async () => {
      if (store.multi_booking_valid) {
        store.currentPageId = 'BOOKING_REVIEW_MULTI'
        await router.push({ name: 'BOOKING_REVIEW_MULTI' })
      }
    }

    const goBack = async () => {
      store.currentPageId = 'MULTI_CITY_INTRO'
      await router.push({ name: 'MULTI_CITY_INTRO' })
    }

    return {
      isValid,
      handlePrimary,
      handleSecond,
      handleCard,
      validateForm,
      submitForm,
      goBack
    }
  }
}
</script>