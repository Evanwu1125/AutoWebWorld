<template>
  <div class="min-h-screen bg-slate-50 font-sans pb-12">
    <header class="bg-white shadow-sm sticky top-0 z-30">
      <div class="max-w-3xl mx-auto px-6 h-16 flex items-center justify-between">
        <div id="back-multi-booking-form" @click="goBack" class="flex items-center gap-2 cursor-pointer text-[#002D5C] hover:text-blue-600 transition-colors">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"/></svg>
          <span class="font-medium">Edit Details</span>
        </div>
        <div class="font-bold text-[#002D5C]">Review Multi-City</div>
        <div class="w-24"></div>
      </div>
    </header>

    <main class="max-w-3xl mx-auto px-6 py-8 space-y-6">
      <div class="bg-white rounded-2xl shadow-sm border border-gray-100 p-8 space-y-6">
        <h2 class="text-xl font-bold text-gray-900 border-b border-gray-100 pb-4">Terms & Conditions</h2>
        
        <label class="flex items-start gap-4 cursor-pointer group">
          <div class="relative flex items-center mt-1">
            <input 
              id="multi-terms-checkbox"
              type="checkbox" 
              @change="handleTerms"
              class="peer h-6 w-6 cursor-pointer appearance-none rounded-lg border border-gray-300 shadow-sm transition-all checked:border-blue-600 checked:bg-blue-600 hover:border-blue-400 focus:ring-2 focus:ring-blue-200" 
            />
            <svg class="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-4 h-4 pointer-events-none opacity-0 peer-checked:opacity-100 text-white" viewBox="0 0 14 14" fill="none">
              <path d="M3 8L6 11L11 3.5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </div>
          <span class="text-gray-600 group-hover:text-gray-900 transition-colors">
            I agree to the Multi-City Booking Terms.
          </span>
        </label>
      </div>

      <button 
        v-if="isReady"
        id="confirm-multi-booking"
        @click="confirmBooking"
        class="w-full py-5 bg-[#00A698] hover:bg-[#008f82] text-white text-xl font-bold rounded-2xl shadow-xl shadow-teal-500/20 transition-all transform hover:-translate-y-1 flex items-center justify-center gap-2"
      >
        Confirm & Pay
      </button>

    </main>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'BOOKING_REVIEW_MULTI',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const isReady = computed(() => store.multi_review_ready)

    const handleTerms = () => {
      store.multi_terms_checked = true
      store.multi_review_ready = true
    }

    const confirmBooking = async () => {
      if (store.multi_review_ready) {
        store.currentPageId = 'BOOKING_COMPLETE_MULTI'
        await router.push({ name: 'BOOKING_COMPLETE_MULTI' })
      }
    }

    const goBack = async () => {
      store.currentPageId = 'BOOKING_FORM_MULTI'
      await router.push({ name: 'BOOKING_FORM_MULTI' })
    }

    return {
      isReady,
      handleTerms,
      confirmBooking,
      goBack
    }
  }
}
</script>