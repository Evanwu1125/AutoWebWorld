<template>
  <div class="min-h-screen bg-gray-50 pb-20">
    <!-- Header -->
    <div class="bg-white px-4 py-4 shadow-sm sticky top-0 z-20">
      <div class="flex items-center justify-between">
        <button 
          id="back-home" 
          @click="goHome"
          class="p-2 -ml-2 rounded-full hover:bg-gray-100 text-gray-600"
        >
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
        </button>
        <h1 class="text-lg font-bold text-gray-900">Accounts</h1>
        <div class="w-10"></div> <!-- Spacer for center alignment -->
      </div>
    </div>

    <!-- Filters Section -->
    <div class="bg-white px-4 py-4 mb-4 border-b border-gray-100">
      <div class="flex flex-wrap gap-3 items-center">
        
        <!-- Checkbox Filter -->
        <button 
          id="filter-hide-zero-balance"
          @click="toggleZeroBalance"
          :class="['px-3 py-1.5 rounded-full text-sm font-medium border transition-colors', hideZeroBalance ? 'bg-blue-100 text-blue-700 border-blue-200' : 'bg-gray-50 text-gray-600 border-gray-200']"
        >
          Hide Zero Balance
        </button>

        <!-- Sort Dropdown -->
        <div class="relative z-30">
          <button 
            id="sort-dropdown"
            @click="showSortMenu = !showSortMenu"
            class="px-3 py-1.5 rounded-full text-sm font-medium border border-gray-200 bg-gray-50 text-gray-700 flex items-center gap-1"
          >
            <span>Sort</span>
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
          </button>

          <div v-if="showSortMenu" class="absolute top-full left-0 mt-2 w-40 bg-white rounded-xl shadow-xl border border-gray-100 overflow-hidden">
            <div 
              id="sort-option-balance-desc" 
              @click="setSort('balance_desc')"
              class="px-4 py-2 hover:bg-gray-50 cursor-pointer text-sm text-gray-700"
            >
              Highest Balance
            </div>
            <div 
              id="sort-option-balance-asc" 
              @click="setSort('balance_asc')"
              class="px-4 py-2 hover:bg-gray-50 cursor-pointer text-sm text-gray-700"
            >
              Lowest Balance
            </div>
            <div 
              id="sort-option-currency" 
              @click="setSort('currency')"
              class="px-4 py-2 hover:bg-gray-50 cursor-pointer text-sm text-gray-700"
            >
              Currency
            </div>
          </div>
        </div>

        <!-- Range Slider -->
        <div class="w-full sm:w-auto flex items-center gap-2 mt-2 sm:mt-0">
          <span class="text-xs font-medium text-gray-500">Min Balance: ${{ minBalance }}</span>
          <input 
            id="balance-slider"
            type="range" 
            min="0" 
            max="10000" 
            step="100"
            v-model="minBalance"
            @input="applySliderFilter"
            class="w-full sm:w-32 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
          />
        </div>
      </div>
    </div>

    <!-- Accounts List -->
    <div id="accounts-list" class="px-4 space-y-3">
      <div 
        v-for="account in filteredAccounts" 
        :key="account.id"
        :class="['bg-white rounded-2xl p-4 shadow-sm border border-transparent hover:border-blue-100 transition-all cursor-pointer flex items-center justify-between', filtersApplied ? 'account-row-filtered' : 'account-row-visible']"
        :data-id="account.id"
        @click="openAccount(account.id)"
      >
        <div class="flex items-center gap-4">
          <div class="w-10 h-10 rounded-full overflow-hidden bg-gray-100 border border-gray-200 shadow-sm">
             <img :src="account.image" alt="Flag" class="w-full h-full object-cover" />
          </div>
          <div>
            <div class="font-bold text-gray-900">{{ account.name }}</div>
            <div class="text-sm text-gray-500">{{ account.type }} • {{ account.currency }}</div>
          </div>
        </div>
        <div class="text-right">
          <div class="font-bold text-gray-900">{{ formatCurrency(account.balance, account.currency) }}</div>
        </div>
      </div>

      <!-- Empty State -->
      <div v-if="filteredAccounts.length === 0" class="text-center py-10">
        <div class="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-3">
          <svg class="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4"></path></svg>
        </div>
        <p class="text-gray-500 font-medium">No accounts found</p>
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
  name: 'ACCOUNTS_DASHBOARD',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const hideZeroBalance = ref(false)
    const minBalance = ref(0)
    const showSortMenu = ref(false)
    const currentSort = ref(null) // 'balance_desc', 'balance_asc', 'currency'
    const filtersApplied = ref(false)

    // Compute filtered and sorted accounts
    const filteredAccounts = computed(() => {
      let result = [...dataStore.accounts]

      // Apply Filters
      if (hideZeroBalance.value) {
        result = result.filter(a => a.balance > 0)
      }

      if (minBalance.value > 0) {
        result = result.filter(a => a.balance >= minBalance.value)
      }

      // Apply Sort
      if (currentSort.value === 'balance_desc') {
        result.sort((a, b) => b.balance - a.balance)
      } else if (currentSort.value === 'balance_asc') {
        result.sort((a, b) => a.balance - b.balance)
      } else if (currentSort.value === 'currency') {
        result.sort((a, b) => a.currency.localeCompare(b.currency))
      }

      return result
    })

    // Formatting helper
    const formatCurrency = (value, currency) => {
      return new Intl.NumberFormat('en-US', { style: 'currency', currency: currency }).format(value)
    }

    // Actions
    const goHome = () => {
      signatureStore.setCurrentPageId('HOME')
      router.push({ name: 'HOME' })
    }

    const toggleZeroBalance = () => {
      hideZeroBalance.value = !hideZeroBalance.value
      filtersApplied.value = true
      signatureStore.accounts_filters_applied = true
    }

    const applySliderFilter = () => {
      filtersApplied.value = true
      signatureStore.accounts_filters_applied = true
    }

    const setSort = (sortType) => {
      currentSort.value = sortType
      showSortMenu.value = false
      filtersApplied.value = true
      signatureStore.accounts_filters_applied = true
    }

    const openAccount = (accountId) => {
      signatureStore.accounts_selected_account_id = accountId
      signatureStore.setCurrentPageId('ACCOUNT_DETAIL')
      
      // Clear filters flag if opening filtered item (as per FSM effect ACT_ACCOUNTS_OPEN_FILTERED_ACCOUNT)
      // But also for open ANY account, so generic logic:
      if (filtersApplied.value) {
          signatureStore.accounts_filters_applied = null // op: clear
      }
      signatureStore.accounts_viewport_anchor_id = null // clear anchor if any

      router.push({ name: 'ACCOUNT_DETAIL' })
    }

    return {
      hideZeroBalance,
      minBalance,
      showSortMenu,
      filteredAccounts,
      filtersApplied,
      formatCurrency,
      goHome,
      toggleZeroBalance,
      applySliderFilter,
      setSort,
      openAccount
    }
  }
}
</script>