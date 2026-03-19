<template>
  <div class="min-h-screen bg-slate-50 font-sans pb-12">
    <header class="bg-[#002D5C] text-white py-4 px-6 shadow-md">
      <div class="max-w-4xl mx-auto flex items-center justify-between">
        <div id="trip-summary-home" @click="goHome" class="flex items-center gap-2 cursor-pointer hover:text-blue-200 transition-colors">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"/></svg>
          <span class="font-bold">Home</span>
        </div>
        <div class="font-bold text-xl">My Trip</div>
        <div class="w-16"></div>
      </div>
    </header>

    <main class="max-w-4xl mx-auto px-6 py-12">
      <div class="bg-white rounded-2xl shadow-xl overflow-hidden">
        <div class="relative h-48 bg-blue-600 overflow-hidden">
           <img src="/images/photo1765608129.jpg" class="w-full h-full object-cover opacity-50 mix-blend-overlay" />
           <div class="absolute inset-0 flex flex-col items-center justify-center text-white p-6 text-center">
             <h1 class="text-3xl font-bold mb-2">Trip to New York</h1>
             <p class="opacity-90">Booking Ref: SKY-8823</p>
           </div>
        </div>

        <div class="p-8 space-y-8">
          <div class="bg-gray-50 rounded-xl p-6 border border-gray-100 flex gap-6 items-start">
            <div class="bg-white p-4 rounded-lg shadow-sm">
               <svg class="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"/></svg>
            </div>
            <div>
              <h3 class="font-bold text-lg text-gray-900 mb-1">Flight Details</h3>
              <p class="text-gray-600">London (LHR) → New York (JFK)</p>
              <p class="text-gray-500 text-sm mt-2">Fri, 22 Oct • 10:00 - 12:55</p>
            </div>
          </div>

          <div class="flex gap-4">
            <button id="back-complete-direct" @click="goBackDirect" class="flex-1 py-3 border border-gray-200 hover:bg-gray-50 rounded-xl font-medium text-gray-600 transition-colors">
              Back to Confirmation
            </button>
            <button id="back-complete-multi" @click="goBackMulti" class="flex-1 py-3 border border-gray-200 hover:bg-gray-50 rounded-xl font-medium text-gray-600 transition-colors">
              Back to Multi Confirmation
            </button>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'TRIP_SUMMARY',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const goHome = async () => {
      store.currentPageId = 'HOME'
      await router.push({ name: 'HOME' })
    }

    const goBackDirect = async () => {
      store.currentPageId = 'BOOKING_COMPLETE_DIRECT'
      await router.push({ name: 'BOOKING_COMPLETE_DIRECT' })
    }

    const goBackMulti = async () => {
      store.currentPageId = 'BOOKING_COMPLETE_MULTI'
      await router.push({ name: 'BOOKING_COMPLETE_MULTI' })
    }

    return {
      goHome,
      goBackDirect,
      goBackMulti
    }
  }
}
</script>