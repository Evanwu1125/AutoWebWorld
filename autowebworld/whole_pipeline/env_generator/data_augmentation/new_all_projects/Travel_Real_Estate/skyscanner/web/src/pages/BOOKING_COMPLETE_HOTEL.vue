<template>
  <div class="min-h-screen bg-slate-50 font-sans flex flex-col items-center justify-center p-6">
    <div class="bg-white rounded-3xl shadow-xl p-8 md:p-12 max-w-lg w-full text-center space-y-8">
      <div class="w-24 h-24 bg-yellow-100 rounded-full flex items-center justify-center mx-auto animate-bounce">
        <svg class="w-12 h-12 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4"/></svg>
      </div>

      <div>
        <h1 class="text-3xl font-bold text-gray-900 mb-2">Reservation Confirmed!</h1>
        <p class="text-gray-600">Your hotel stay is locked in.</p>
      </div>

      <div class="bg-gray-50 rounded-xl p-6 border border-gray-100">
        <div class="text-sm font-bold text-gray-400 uppercase tracking-wide mb-1">Confirmation Number</div>
        <div class="text-3xl font-mono font-bold text-[#002D5C] tracking-widest">SKY-HOTEL-77</div>
      </div>

      <div class="flex flex-col gap-4">
        <button 
          id="hotel-complete-home"
          @click="goHome"
          class="w-full py-4 bg-[#0770E3] hover:bg-[#0660C3] text-white font-bold rounded-xl shadow-lg shadow-blue-600/20 transition-all transform hover:-translate-y-0.5"
        >
          Back to Home
        </button>
        <div class="pt-4 border-t border-gray-100">
           <button id="back-hotel-details" @click="goBack" class="text-sm text-gray-400 hover:text-gray-600 underline">Back (Debug)</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'BOOKING_COMPLETE_HOTEL',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const goHome = async () => {
      store.currentPageId = 'HOME'
      await router.push({ name: 'HOME' })
    }

    const goBack = async () => {
      store.currentPageId = 'HOTEL_DETAILS'
      await router.push({ name: 'HOTEL_DETAILS' })
    }

    return {
      goHome,
      goBack
    }
  }
}
</script>