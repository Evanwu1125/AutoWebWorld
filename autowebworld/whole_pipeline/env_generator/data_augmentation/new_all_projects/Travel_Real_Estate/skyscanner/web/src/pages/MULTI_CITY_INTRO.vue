<template>
  <div class="min-h-screen bg-slate-50 font-sans pb-12">
    <header class="bg-white shadow-sm sticky top-0 z-30">
      <div class="max-w-4xl mx-auto px-6 h-16 flex items-center justify-between">
        <div id="back-multi-results" @click="goBack" class="flex items-center gap-2 cursor-pointer text-[#002D5C] hover:text-blue-600 transition-colors">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"/></svg>
          <span class="font-medium">Back to Results</span>
        </div>
        <div class="font-bold text-[#002D5C]">Trip Itinerary</div>
        <div class="w-24"></div>
      </div>
    </header>

    <main class="max-w-4xl mx-auto px-6 py-8 space-y-6">
      <div class="bg-white rounded-2xl shadow-xl overflow-hidden p-8">
        <h2 class="text-2xl font-bold text-gray-900 mb-2">Your Multi-City Trip</h2>
        <p class="text-gray-500 mb-8">Review the flight segments below.</p>

        <div class="border rounded-xl border-gray-200 overflow-hidden">
          <div id="expand-segments" @click="viewSegments" class="bg-gray-50 p-4 cursor-pointer flex justify-between items-center hover:bg-gray-100 transition-colors">
            <span class="font-bold text-gray-700">Flight Segments Details</span>
            <svg :class="{'rotate-180': segmentsViewed}" class="w-5 h-5 text-gray-500 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
          </div>
          
          <div v-if="segmentsViewed" class="p-6 bg-white space-y-6">
            <div class="flex gap-4 items-start">
              <div class="w-8 h-8 rounded-full bg-blue-600 text-white flex items-center justify-center font-bold text-xs shrink-0">1</div>
              <div>
                <h4 class="font-bold text-gray-900">London (LHR) to New York (JFK)</h4>
                <p class="text-sm text-gray-500">Fri, 22 Oct • 10:00 - 12:55</p>
              </div>
            </div>
            <div class="w-px h-8 bg-gray-200 ml-4"></div>
             <div class="flex gap-4 items-start">
              <div class="w-8 h-8 rounded-full bg-blue-600 text-white flex items-center justify-center font-bold text-xs shrink-0">2</div>
              <div>
                <h4 class="font-bold text-gray-900">New York (JFK) to Los Angeles (LAX)</h4>
                <p class="text-sm text-gray-500">Mon, 25 Oct • 14:00 - 17:30</p>
              </div>
            </div>
          </div>
        </div>

        <div class="mt-8 flex justify-end">
          <button 
            v-if="segmentsViewed"
            id="continue-multi-booking"
            @click="goToBooking"
            class="px-8 py-4 bg-[#00A698] hover:bg-[#008f82] text-white font-bold rounded-xl shadow-lg transition-all"
          >
            Continue to Booking
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

export default {
  name: 'MULTI_CITY_INTRO',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const segmentsViewed = computed(() => store.segments_summary_viewed)

    const viewSegments = () => {
      store.segments_summary_viewed = true
    }

    const goToBooking = async () => {
      if (store.segments_summary_viewed) {
        store.currentPageId = 'BOOKING_FORM_MULTI'
        await router.push({ name: 'BOOKING_FORM_MULTI' })
      }
    }

    const goBack = async () => {
      store.currentPageId = 'MULTI_CITY_RESULTS'
      await router.push({ name: 'MULTI_CITY_RESULTS' })
    }

    return {
      segmentsViewed,
      viewSegments,
      goToBooking,
      goBack
    }
  }
}
</script>