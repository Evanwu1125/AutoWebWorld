<template>
  <div class="min-h-screen bg-slate-50 font-sans">
    <header class="bg-[#002D5C] text-white py-4 px-6 shadow-md">
      <div class="max-w-7xl mx-auto flex items-center justify-between">
        <div id="back-flights-search" @click="goBack" class="flex items-center gap-2 cursor-pointer hover:text-blue-200 transition-colors">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"/></svg>
          <span class="font-medium">Back to Search</span>
        </div>
        <div class="font-bold text-xl">Multi-City Search</div>
        <div class="w-24"></div>
      </div>
    </header>

    <main class="max-w-4xl mx-auto px-6 py-12">
      <div class="bg-white rounded-2xl shadow-xl overflow-hidden p-8">
        <h2 class="text-2xl font-bold text-gray-900 mb-6">Build Your Trip</h2>
        
        <div class="space-y-8">
          <!-- Flight 1 -->
          <div class="bg-gray-50 p-6 rounded-xl border border-gray-100">
            <h3 class="text-sm font-bold text-gray-500 uppercase tracking-wide mb-4 flex items-center gap-2">
              <span class="bg-blue-600 text-white w-6 h-6 rounded-full flex items-center justify-center text-xs">1</span>
              Flight 1
            </h3>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <input 
                id="leg1-origin-input"
                type="text" 
                @input="handleLeg1Input"
                class="block w-full px-4 py-3 bg-white border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
                placeholder="From (e.g. London)"
              />
              <div class="px-4 py-3 bg-white border border-gray-200 rounded-lg text-gray-400 cursor-not-allowed">To (Select Destination)</div>
            </div>
          </div>

          <!-- Flight 2 -->
          <div class="bg-gray-50 p-6 rounded-xl border border-gray-100">
            <h3 class="text-sm font-bold text-gray-500 uppercase tracking-wide mb-4 flex items-center gap-2">
              <span class="bg-blue-600 text-white w-6 h-6 rounded-full flex items-center justify-center text-xs">2</span>
              Flight 2
            </h3>
             <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div class="px-4 py-3 bg-white border border-gray-200 rounded-lg text-gray-400 cursor-not-allowed">From (Previous Destination)</div>
              <input 
                id="leg2-destination-input"
                type="text" 
                @input="handleLeg2Input"
                 class="block w-full px-4 py-3 bg-white border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
                placeholder="To (e.g. New York)"
              />
            </div>
          </div>

          <div class="flex gap-4 pt-4 border-t border-gray-100">
             <button 
              id="multi-validate-button"
              @click="validateMulti"
              class="flex-1 py-4 bg-gray-100 hover:bg-gray-200 text-gray-800 font-bold rounded-xl transition-colors"
            >
              Verify Route
            </button>
            <button 
              v-if="isValid"
              id="search-multicity-button"
              @click="submitMultiSearch"
              class="flex-[2] py-4 bg-[#0770E3] hover:bg-[#0660C3] text-white font-bold rounded-xl shadow-lg shadow-blue-600/20 transition-all transform hover:-translate-y-0.5"
            >
              Search Multi-City Flights
            </button>
          </div>
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
  name: 'MULTI_CITY_SEARCH',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const isValid = computed(() => store.multi_city_valid)

    const handleLeg1Input = () => store.leg1_filled = true
    const handleLeg2Input = () => store.leg2_filled = true

    const validateMulti = () => {
      if (store.leg1_filled && store.leg2_filled) {
        store.multi_city_valid = true
      }
    }

    const submitMultiSearch = async () => {
      if (store.multi_city_valid) {
        store.currentPageId = 'MULTI_CITY_RESULTS'
        await router.push({ name: 'MULTI_CITY_RESULTS' })
      }
    }

    const goBack = async () => {
      store.currentPageId = 'FLIGHTS_SEARCH'
      await router.push({ name: 'FLIGHTS_SEARCH' })
    }

    return {
      isValid,
      handleLeg1Input,
      handleLeg2Input,
      validateMulti,
      submitMultiSearch,
      goBack
    }
  }
}
</script>