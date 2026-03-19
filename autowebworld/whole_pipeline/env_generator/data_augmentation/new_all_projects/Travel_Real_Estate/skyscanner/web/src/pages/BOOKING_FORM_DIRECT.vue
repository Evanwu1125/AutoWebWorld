<template>
  <div class="min-h-screen bg-slate-50 font-sans pb-12">
    <header class="bg-white shadow-sm sticky top-0 z-30">
      <div class="max-w-3xl mx-auto px-6 h-16 flex items-center justify-between">
        <div id="back-flight-details" @click="goBack" class="flex items-center gap-2 cursor-pointer text-[#002D5C] hover:text-blue-600 transition-colors">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"/></svg>
          <span class="font-medium">Back to Details</span>
        </div>
        <div class="font-bold text-[#002D5C]">Passenger Details</div>
        <div class="w-24"></div>
      </div>
    </header>

    <main class="max-w-3xl mx-auto px-6 py-8 space-y-6">
      <div class="bg-white rounded-2xl shadow-sm border border-gray-100 p-8 space-y-6">
        <h2 class="text-xl font-bold text-gray-900 border-b border-gray-100 pb-4">Who's flying?</h2>
        
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label class="block text-sm font-bold text-gray-700 mb-2">First Name</label>
            <input 
              id="first-name-input"
              type="text" 
              @input="handleFirstName"
              class="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none transition-all"
              placeholder="e.g. John"
            />
          </div>
          <div>
            <label class="block text-sm font-bold text-gray-700 mb-2">Last Name</label>
            <input 
              id="last-name-input"
              type="text" 
              @input="handleLastName"
              class="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none transition-all"
              placeholder="e.g. Doe"
            />
          </div>
        </div>
      </div>

      <div class="bg-white rounded-2xl shadow-sm border border-gray-100 p-8 space-y-6">
        <h2 class="text-xl font-bold text-gray-900 border-b border-gray-100 pb-4">Contact Info</h2>
        <div>
          <label class="block text-sm font-bold text-gray-700 mb-2">Email Address</label>
          <input 
            id="contact-email-input"
            type="email" 
            @input="handleEmail"
            class="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none transition-all"
            placeholder="Where should we send your confirmation?"
          />
        </div>
      </div>

      <div class="bg-white rounded-2xl shadow-sm border border-gray-100 p-8 space-y-6">
        <h2 class="text-xl font-bold text-gray-900 border-b border-gray-100 pb-4">Payment</h2>
        <div>
          <label class="block text-sm font-bold text-gray-700 mb-2">Card Number</label>
          <div class="relative">
            <input 
              id="card-number-input"
              type="text" 
              @input="handleCard"
              class="w-full pl-12 pr-4 py-3 bg-gray-50 border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none transition-all"
              placeholder="0000 0000 0000 0000"
            />
            <svg class="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z"/></svg>
          </div>
        </div>
      </div>

      <button 
        id="booking-validate-button"
        @click="validateForm"
        class="w-full py-4 bg-gray-100 hover:bg-gray-200 text-gray-800 font-bold rounded-xl transition-colors"
      >
        Verify Details
      </button>

      <button 
        v-if="isValid"
        id="continue-to-review"
        @click="submitForm"
        class="w-full py-5 bg-[#00A698] hover:bg-[#008f82] text-white text-xl font-bold rounded-2xl shadow-xl shadow-teal-500/20 transition-all transform hover:-translate-y-1 flex items-center justify-center gap-2"
      >
        Review Booking
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"/></svg>
      </button>

    </main>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'BOOKING_FORM_DIRECT',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const isValid = computed(() => store.booking_form_valid)

    const handleFirstName = () => store.passenger_first_name_entered = true
    const handleLastName = () => store.passenger_last_name_entered = true
    const handleEmail = () => store.contact_email_entered = true
    const handleCard = () => store.payment_card_entered = true

    const validateForm = () => {
      if (store.passenger_first_name_entered && 
          store.passenger_last_name_entered && 
          store.contact_email_entered && 
          store.payment_card_entered) {
        store.booking_form_valid = true
      }
    }

    const submitForm = async () => {
      if (store.booking_form_valid) {
        store.currentPageId = 'BOOKING_REVIEW_DIRECT'
        await router.push({ name: 'BOOKING_REVIEW_DIRECT' })
      }
    }

    const goBack = async () => {
      store.currentPageId = 'FLIGHT_DETAILS'
      await router.push({ name: 'FLIGHT_DETAILS' })
    }

    return {
      isValid,
      handleFirstName,
      handleLastName,
      handleEmail,
      handleCard,
      validateForm,
      submitForm,
      goBack
    }
  }
}
</script>