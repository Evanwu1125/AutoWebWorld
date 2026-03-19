<template>
  <div class="min-h-screen bg-gray-50 pb-20">
    <!-- Header -->
    <div class="bg-white px-4 py-4 shadow-sm sticky top-0 z-20 flex items-center justify-between">
      <button 
        id="back-home-exchange" 
        @click="goHome"
        class="p-2 -ml-2 rounded-full hover:bg-gray-100 text-gray-600"
      >
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
      </button>
      <h1 class="text-lg font-bold text-gray-900">Exchange Rates</h1>
      <div class="w-10"></div>
    </div>

    <!-- Filters -->
    <div class="bg-white px-4 py-4 mb-4 border-b border-gray-100">
      <div class="flex flex-wrap gap-3 items-center">
        <!-- Majors Only Filter -->
        <button 
          id="filter-majors-only"
          @click="toggleMajorsOnly"
          :class="['px-3 py-1.5 rounded-full text-sm font-medium border transition-colors', majorsOnly ? 'bg-green-100 text-green-700 border-green-200' : 'bg-gray-50 text-gray-600 border-gray-200']"
        >
          Majors Only
        </button>

        <!-- Sort Dropdown -->
        <div class="relative z-30">
          <button 
            id="sort-dropdown-exchange"
            @click="showSortMenu = !showSortMenu"
            class="px-3 py-1.5 rounded-full text-sm font-medium border border-gray-200 bg-gray-50 text-gray-700 flex items-center gap-1"
          >
            <span>Sort</span>
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
          </button>

          <div v-if="showSortMenu" class="absolute top-full left-0 mt-2 w-40 bg-white rounded-xl shadow-xl border border-gray-100 overflow-hidden">
            <div 
              id="sort-option-best-rate-desc" 
              @click="setSort('best_rate')"
              class="px-4 py-2 hover:bg-gray-50 cursor-pointer text-sm text-gray-700"
            >
              Best Rate
            </div>
            <div 
              id="sort-option-a-z" 
              @click="setSort('a_to_z')"
              class="px-4 py-2 hover:bg-gray-50 cursor-pointer text-sm text-gray-700"
            >
              A to Z
            </div>
            <div 
              id="sort-option-z-a" 
              @click="setSort('z_to_a')"
              class="px-4 py-2 hover:bg-gray-50 cursor-pointer text-sm text-gray-700"
            >
              Z to A
            </div>
          </div>
        </div>

        <!-- Slider (Rate Change) -->
        <div class="w-full sm:w-auto flex items-center gap-2 mt-2 sm:mt-0">
          <span class="text-xs font-medium text-gray-500">Min Change: {{ minChange }}%</span>
          <input 
            id="rate-change-slider"
            type="range" 
            min="0" 
            max="5" 
            step="0.1"
            v-model="minChange"
            @input="applySliderFilter"
            class="w-full sm:w-32 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-green-600"
          />
        </div>
      </div>
    </div>

    <!-- Pairs List -->
    <div id="exchange-list" class="px-4 space-y-3">
      <div 
        v-for="pair in filteredPairs" 
        :key="pair.id"
        :class="['bg-white rounded-2xl p-4 shadow-sm border border-transparent hover:border-green-100 transition-all cursor-pointer flex items-center justify-between',`data-id-${pair.id}`, filtersApplied ? 'pair-row-filtered' : 'pair-row-visible']"
        :data-id="pair.id"
        @click="openPair(pair.id)"
      >
        <div class="flex items-center gap-4">
          <div class="w-12 h-8 rounded-md overflow-hidden bg-gray-100 shadow-sm relative border border-gray-200">
             <!-- Simulating split flag look -->
             <img :src="pair.image" class="w-full h-full object-cover" />
          </div>
          <div>
            <div class="font-bold text-gray-900">{{ pair.from }} → {{ pair.to }}</div>
            <div :class="['text-xs font-bold', pair.change >= 0 ? 'text-green-500' : 'text-red-500']">
              {{ pair.change >= 0 ? '+' : '' }}{{ pair.change }}% today
            </div>
          </div>
        </div>
        <div class="text-right">
          <div class="font-mono font-bold text-gray-900 text-lg">{{ pair.rate }}</div>
        </div>
      </div>

      <!-- Empty State -->
      <div v-if="filteredPairs.length === 0" class="text-center py-10">
        <p class="text-gray-500 font-medium">No pairs found</p>
      </div>
    </div>

  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'EXCHANGE_DASHBOARD',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const majorsOnly = ref(false)
    const minChange = ref(0)
    const showSortMenu = ref(false)
    const currentSort = ref(null)
    const filtersApplied = ref(false)

    const filteredPairs = computed(() => {
      let result = [...dataStore.exchangePairs]

      if (majorsOnly.value) {
        result = result.filter(p => p.isMajor)
      }

      if (minChange.value > 0) {
        result = result.filter(p => Math.abs(p.change) >= minChange.value)
      }

      if (currentSort.value === 'best_rate') {
        result.sort((a, b) => b.rate - a.rate)
      } else if (currentSort.value === 'a_to_z') {
        result.sort((a, b) => (a.from + a.to).localeCompare(b.from + b.to))
      } else if (currentSort.value === 'z_to_a') {
        result.sort((a, b) => (b.from + b.to).localeCompare(a.from + a.to))
      }

      return result
    })

    const goHome = () => {
      signatureStore.setCurrentPageId('HOME')
      router.push({ name: 'HOME' })
    }

    const toggleMajorsOnly = () => {
      majorsOnly.value = !majorsOnly.value
      filtersApplied.value = true
      signatureStore.exchange_filters_applied = true
    }

    const applySliderFilter = () => {
      filtersApplied.value = true
      signatureStore.exchange_filters_applied = true
    }

    const setSort = (type) => {
      currentSort.value = type
      showSortMenu.value = false
      filtersApplied.value = true
      signatureStore.exchange_filters_applied = true
    }

    const openPair = (id) => {
      signatureStore.exchange_selected_pair_id = id
      
      if (filtersApplied.value) {
        signatureStore.exchange_filters_applied = null
      }
      signatureStore.exchange_viewport_anchor_id = null

      signatureStore.setCurrentPageId('EXCHANGE_FORM')
      router.push({ name: 'EXCHANGE_FORM' })
    }

    return {
      majorsOnly,
      minChange,
      showSortMenu,
      filteredPairs,
      filtersApplied,
      goHome,
      toggleMajorsOnly,
      applySliderFilter,
      setSort,
      openPair
    }
  }
}
</script>