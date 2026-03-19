<template>
  <div class="min-h-screen bg-gray-50 pb-20">
    <!-- Header -->
    <div class="bg-white px-4 py-4 shadow-sm sticky top-0 z-20 flex items-center justify-between">
      <button 
        id="back-home-cards" 
        @click="goHome"
        class="p-2 -ml-2 rounded-full hover:bg-gray-100 text-gray-600"
      >
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
      </button>
      <h1 class="text-lg font-bold text-gray-900">My Cards</h1>
      <button class="p-2 -mr-2 text-blue-600 font-bold text-sm bg-blue-50 rounded-lg hover:bg-blue-100 transition-colors">
        + Add
      </button>
    </div>

    <!-- Filters -->
    <div class="bg-white px-4 py-4 mb-4 border-b border-gray-100">
      <div class="flex flex-wrap gap-3 items-center">
        <!-- Active Only Filter -->
        <button 
          id="filter-active-only"
          @click="toggleActiveOnly"
          :class="['px-3 py-1.5 rounded-full text-sm font-medium border transition-colors', activeOnly ? 'bg-blue-100 text-blue-700 border-blue-200' : 'bg-gray-50 text-gray-600 border-gray-200']"
        >
          Active Only
        </button>

        <!-- Sort Dropdown -->
        <div class="relative z-30">
          <button 
            id="sort-dropdown-cards"
            @click="showSortMenu = !showSortMenu"
            class="px-3 py-1.5 rounded-full text-sm font-medium border border-gray-200 bg-gray-50 text-gray-700 flex items-center gap-1"
          >
            <span>Sort</span>
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
          </button>

          <div v-if="showSortMenu" class="absolute top-full left-0 mt-2 w-40 bg-white rounded-xl shadow-xl border border-gray-100 overflow-hidden">
            <div 
              id="sort-option-created-desc" 
              @click="setSort('created')"
              class="px-4 py-2 hover:bg-gray-50 cursor-pointer text-sm text-gray-700"
            >
              Date Created
            </div>
            <div 
              id="sort-option-nickname" 
              @click="setSort('nickname')"
              class="px-4 py-2 hover:bg-gray-50 cursor-pointer text-sm text-gray-700"
            >
              Nickname
            </div>
          </div>
        </div>

        <!-- Slider (Limit Range) -->
        <div class="w-full sm:w-auto flex items-center gap-2 mt-2 sm:mt-0">
          <span class="text-xs font-medium text-gray-500">Min Limit: ${{ minLimit }}</span>
          <input 
            id="limit-range-slider"
            type="range" 
            min="0" 
            max="10000" 
            step="100"
            v-model="minLimit"
            @input="applySliderFilter"
            class="w-full sm:w-32 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
          />
        </div>
      </div>
    </div>

    <!-- Cards List -->
    <div id="cards-list" class="px-4 space-y-4">
      <div
        v-for="card in filteredCards"
        :key="card.id"
        :class="['group relative h-48 rounded-2xl p-6 shadow-lg text-white transition-all transform hover:scale-[1.02] cursor-pointer flex flex-col justify-between overflow-hidden', `data-id-${card.id}`, filtersApplied ? 'card-row-filtered' : 'card-row-visible']"
        :data-id="card.id"
        @click="openCard(card.id)"
      >
        <!-- Background Image -->
        <div class="absolute inset-0 z-0">
          <img :src="card.image" class="w-full h-full object-cover" />
          <div class="absolute inset-0 bg-black/20 group-hover:bg-black/10 transition-colors"></div>
        </div>

        <!-- Content -->
        <div class="relative z-10 flex justify-between items-start">
          <div>
             <div class="font-bold text-lg tracking-wide">{{ card.nickname }}</div>
             <div class="text-sm opacity-80">{{ card.type }}</div>
          </div>
          <div class="font-bold italic opacity-90">{{ card.scheme }}</div>
        </div>

        <div class="relative z-10">
          <div class="text-xl font-mono tracking-widest mb-1">•••• •••• •••• {{ card.last4 }}</div>
          <div class="flex justify-between items-end">
            <div class="text-sm opacity-80">Exp {{ card.expiry }}</div>
            <div 
              :class="['px-2 py-0.5 rounded text-xs font-bold uppercase tracking-wider', card.status === 'Active' ? 'bg-green-400 text-green-900' : 'bg-red-400 text-red-900']"
            >
              {{ card.status }}
            </div>
          </div>
        </div>
      </div>

      <!-- Empty State -->
      <div v-if="filteredCards.length === 0" class="text-center py-10">
        <p class="text-gray-500 font-medium">No cards found</p>
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
  name: 'CARDS_LIST',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const activeOnly = ref(false)
    const minLimit = ref(0)
    const showSortMenu = ref(false)
    const currentSort = ref(null)
    const filtersApplied = ref(false)

    const filteredCards = computed(() => {
      let result = [...dataStore.cards]

      if (activeOnly.value) {
        result = result.filter(c => c.status === 'Active')
      }

      if (minLimit.value > 0) {
        result = result.filter(c => c.limit >= minLimit.value)
      }

      if (currentSort.value === 'created') {
        result.sort((a, b) => new Date(b.created) - new Date(a.created))
      } else if (currentSort.value === 'nickname') {
        result.sort((a, b) => a.nickname.localeCompare(b.nickname))
      }

      return result
    })

    const goHome = () => {
      signatureStore.setCurrentPageId('HOME')
      router.push({ name: 'HOME' })
    }

    const toggleActiveOnly = () => {
      activeOnly.value = !activeOnly.value
      filtersApplied.value = true
      signatureStore.cards_filters_applied = true
    }

    const applySliderFilter = () => {
      filtersApplied.value = true
      signatureStore.cards_filters_applied = true
    }

    const setSort = (type) => {
      currentSort.value = type
      showSortMenu.value = false
      filtersApplied.value = true
      signatureStore.cards_filters_applied = true
    }

    const openCard = (id) => {
      signatureStore.cards_selected_card_id = id
      
      if (filtersApplied.value) {
        signatureStore.cards_filters_applied = null
      }
      signatureStore.cards_viewport_anchor_id = null

      signatureStore.setCurrentPageId('CARD_DETAIL')
      router.push({ name: 'CARD_DETAIL' })
    }

    return {
      activeOnly,
      minLimit,
      showSortMenu,
      filteredCards,
      filtersApplied,
      goHome,
      toggleActiveOnly,
      applySliderFilter,
      setSort,
      openCard
    }
  }
}
</script>