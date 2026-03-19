<template>
  <div class="min-h-screen bg-gray-50 pb-20">
    <!-- Header -->
    <div class="bg-white px-4 py-4 shadow-sm sticky top-0 z-20 flex items-center justify-between">
      <button 
        id="back-accounts" 
        @click="goBack"
        class="p-2 -ml-2 rounded-full hover:bg-gray-100 text-gray-600"
      >
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
      </button>
      <h1 class="text-lg font-bold text-gray-900">Transfers</h1>
      <button 
        id="cta-new-transfer"
        @click="goToNewTransfer"
        class="p-2 -mr-2 text-blue-600 font-bold text-sm bg-blue-50 rounded-lg hover:bg-blue-100 transition-colors"
      >
        + New
      </button>
    </div>

    <!-- Filters -->
    <div class="bg-white px-4 py-4 mb-4 border-b border-gray-100">
      <div class="flex flex-wrap gap-3 items-center">
        <!-- Favorites Filter -->
        <button 
          id="filter-favorites-only"
          @click="toggleFavorites"
          :class="['px-3 py-1.5 rounded-full text-sm font-medium border transition-colors flex items-center gap-1', favoritesOnly ? 'bg-yellow-50 text-yellow-700 border-yellow-200' : 'bg-gray-50 text-gray-600 border-gray-200']"
        >
          <svg class="w-4 h-4" :class="favoritesOnly ? 'text-yellow-500' : 'text-gray-400'" fill="currentColor" viewBox="0 0 20 20"><path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"></path></svg>
          Favorites
        </button>

        <!-- Sort Dropdown -->
        <div class="relative z-30">
          <button 
            id="sort-dropdown-payments"
            @click="showSortMenu = !showSortMenu"
            class="px-3 py-1.5 rounded-full text-sm font-medium border border-gray-200 bg-gray-50 text-gray-700 flex items-center gap-1"
          >
            <span>Sort</span>
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
          </button>

          <div v-if="showSortMenu" class="absolute top-full left-0 mt-2 w-40 bg-white rounded-xl shadow-xl border border-gray-100 overflow-hidden">
            <div 
              id="sort-option-recent" 
              @click="setSort('recent')"
              class="px-4 py-2 hover:bg-gray-50 cursor-pointer text-sm text-gray-700"
            >
              Recent
            </div>
            <div 
              id="sort-option-name" 
              @click="setSort('name')"
              class="px-4 py-2 hover:bg-gray-50 cursor-pointer text-sm text-gray-700"
            >
              Name (A-Z)
            </div>
            <!-- 'Amount' sort is weird for beneficiaries list but required by FSM - maybe last transfer amount? 
                 We will mock it or just sort by ID/random for now as data doesn't have amounts directly -->
            <div 
              id="sort-option-amount" 
              @click="setSort('amount')"
              class="px-4 py-2 hover:bg-gray-50 cursor-pointer text-sm text-gray-700"
            >
              Amount
            </div>
          </div>
        </div>

        <!-- Slider (Amount Filter?) FSM says 'amount-slider' -->
        <div class="w-full sm:w-auto flex items-center gap-2 mt-2 sm:mt-0">
          <span class="text-xs font-medium text-gray-500">Min Amount: ${{ minAmount }}</span>
          <input 
            id="amount-slider"
            type="range" 
            min="0" 
            max="5000" 
            step="100"
            v-model="minAmount"
            @input="applySliderFilter"
            class="w-full sm:w-32 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
          />
        </div>
      </div>
    </div>

    <!-- Beneficiaries List -->
    <div id="beneficiaries-list" class="px-4 space-y-3">
      <div 
        v-for="person in filteredBeneficiaries" 
        :key="person.id"
        :class="['bg-white rounded-2xl p-4 shadow-sm border border-transparent hover:border-blue-100 transition-all cursor-pointer flex items-center justify-between', filtersApplied ? 'beneficiary-row-filtered' : 'beneficiary-row-visible']"
        :data-id="person.id"
        @click="openBeneficiary(person.id)"
      >
        <div class="flex items-center gap-4">
          <div class="w-12 h-12 rounded-full overflow-hidden bg-gray-100 border border-gray-200 shadow-sm relative">
             <img :src="person.image" alt="Avatar" class="w-full h-full object-cover" />
             <div v-if="person.isFavorite" class="absolute bottom-0 right-0 bg-yellow-400 rounded-full p-0.5 border-2 border-white w-4 h-4 flex items-center justify-center">
               <svg class="w-2 h-2 text-white" fill="currentColor" viewBox="0 0 20 20"><path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"></path></svg>
             </div>
          </div>
          <div>
            <div class="font-bold text-gray-900">{{ person.name }}</div>
            <div class="text-sm text-gray-500">{{ person.bank }} • {{ person.accountNumber }}</div>
          </div>
        </div>
        <div class="text-gray-400">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path></svg>
        </div>
      </div>

      <!-- Empty State -->
      <div v-if="filteredBeneficiaries.length === 0" class="text-center py-10">
        <p class="text-gray-500 font-medium">No beneficiaries found</p>
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
  name: 'PAYMENTS_LIST',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const favoritesOnly = ref(false)
    const minAmount = ref(0) // Assuming this filters based on some historical transfer amount? 
                             // FSM implies it's a slider. We'll map it to nothing real or mock "last transfer" if needed.
                             // For realism, let's assume we filter users who received > X amount recently, or just ignore logic if data missing.
                             // Actually dataStore beneficiaries doesn't have amount. We'll simulate filter logic or just return all if min=0.
    const showSortMenu = ref(false)
    const currentSort = ref(null)
    const filtersApplied = ref(false)

    const filteredBeneficiaries = computed(() => {
      let result = [...dataStore.beneficiaries]

      if (favoritesOnly.value) {
        result = result.filter(b => b.isFavorite)
      }

      // Slider filter simulation (randomly filter out some for demo if value > 0)
      // Since we don't have amounts in beneficiary data, we'll skip logic or add mock 'totalSent' field
      if (minAmount.value > 0) {
         // Mock logic: filter based on ID length or char code to simulate variety
         result = result.filter(b => b.name.length * 100 > minAmount.value) 
      }

      if (currentSort.value === 'name') {
        result.sort((a, b) => a.name.localeCompare(b.name))
      } else if (currentSort.value === 'recent') {
        // Mock recent: reverse order of ID
        result.reverse()
      } else if (currentSort.value === 'amount') {
        // Mock amount sort
        result.sort((a, b) => a.name.length - b.name.length)
      }

      return result
    })

    const goBack = () => {
      signatureStore.setCurrentPageId('ACCOUNTS_DASHBOARD')
      router.push({ name: 'ACCOUNTS_DASHBOARD' })
    }

    const goToNewTransfer = () => {
      signatureStore.setCurrentPageId('TRANSFER_FORM')
      router.push({ name: 'TRANSFER_FORM' })
    }

    const toggleFavorites = () => {
      favoritesOnly.value = !favoritesOnly.value
      filtersApplied.value = true
      signatureStore.payments_filters_applied = true
    }

    const applySliderFilter = () => {
      filtersApplied.value = true
      signatureStore.payments_filters_applied = true
    }

    const setSort = (type) => {
      currentSort.value = type
      showSortMenu.value = false
      filtersApplied.value = true
      signatureStore.payments_filters_applied = true
    }

    const openBeneficiary = (id) => {
      signatureStore.payments_selected_beneficiary_id = id
      
      if (filtersApplied.value) {
        signatureStore.payments_filters_applied = null
      }
      signatureStore.payments_viewport_anchor_id = null

      signatureStore.setCurrentPageId('BENEFICIARY_DETAIL')
      router.push({ name: 'BENEFICIARY_DETAIL' })
    }

    return {
      favoritesOnly,
      minAmount,
      showSortMenu,
      filteredBeneficiaries,
      filtersApplied,
      goBack,
      goToNewTransfer,
      toggleFavorites,
      applySliderFilter,
      setSort,
      openBeneficiary
    }
  }
}
</script>