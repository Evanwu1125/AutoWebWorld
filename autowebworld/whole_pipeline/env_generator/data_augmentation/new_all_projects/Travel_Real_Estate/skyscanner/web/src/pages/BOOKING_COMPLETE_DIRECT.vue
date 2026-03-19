<template>
  <div class="min-h-screen bg-slate-50 font-sans flex flex-col items-center justify-center p-6">
    <div class="bg-white rounded-3xl shadow-xl p-8 md:p-12 max-w-lg w-full text-center space-y-8">
      <div class="w-24 h-24 bg-green-100 rounded-full flex items-center justify-center mx-auto animate-bounce">
        <svg class="w-12 h-12 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>
      </div>

      <div>
        <h1 class="text-3xl font-bold text-gray-900 mb-2">Booking Confirmed!</h1>
        <p class="text-gray-600">Your flight has been successfully booked. A confirmation email is on its way.</p>
      </div>

      <div class="bg-gray-50 rounded-xl p-6 border border-gray-100">
        <div class="text-sm font-bold text-gray-400 uppercase tracking-wide mb-1">Booking Reference</div>
        <div class="text-3xl font-mono font-bold text-[#002D5C] tracking-widest">SKY-8823</div>
      </div>

      <div class="flex flex-col gap-4">
        <button 
          id="view-trip-button"
          @click="viewTrip"
          class="w-full py-4 bg-[#0770E3] hover:bg-[#0660C3] text-white font-bold rounded-xl shadow-lg shadow-blue-600/20 transition-all transform hover:-translate-y-0.5"
        >
          View Trip Summary
        </button>
        <button 
          id="go-home"
          @click="goHome"
          class="w-full py-4 bg-white border-2 border-gray-100 hover:border-gray-300 text-gray-700 font-bold rounded-xl transition-all"
        >
          Back to Home
        </button>
        <div class="pt-4 border-t border-gray-100">
           <button id="back-review" @click="goBack" class="text-sm text-gray-400 hover:text-gray-600 underline">Back (Debug)</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'BOOKING_COMPLETE_DIRECT',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const viewTrip = async () => {
      store.currentPageId = 'TRIP_SUMMARY'
      await router.push({ name: 'TRIP_SUMMARY' })
    }

    const goHome = async () => {
      store.currentPageId = 'HOME'
      await router.push({ name: 'HOME' })
    }

    const goBack = async () => {
      store.currentPageId = 'BOOKING_REVIEW_DIRECT'
      await router.push({ name: 'BOOKING_REVIEW_DIRECT' })
    }

    return {
      viewTrip,
      goHome,
      goBack
    }
  }
}
</script>