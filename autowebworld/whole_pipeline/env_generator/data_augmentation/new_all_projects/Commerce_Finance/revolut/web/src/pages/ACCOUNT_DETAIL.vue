<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <!-- Top Nav -->
    <div class="bg-white px-4 py-4 shadow-sm sticky top-0 z-20 flex items-center justify-between">
      <button 
        id="back-accounts" 
        @click="goBack"
        class="p-2 -ml-2 rounded-full hover:bg-gray-100 text-gray-600"
      >
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
      </button>
      <h1 class="text-lg font-bold text-gray-900">Details</h1>
      <div class="w-10"></div>
    </div>

    <!-- Account Info -->
    <div v-if="account" class="flex-1 p-4 flex flex-col items-center">
      <div class="w-20 h-20 rounded-full overflow-hidden shadow-md mb-4 bg-white p-1">
         <img :src="account.image" class="w-full h-full object-cover rounded-full" alt="Account" />
      </div>
      
      <h2 class="text-2xl font-bold text-gray-900 mb-1">{{ account.name }}</h2>
      <p class="text-gray-500 mb-6">{{ account.type }} • {{ account.currency }}</p>
      
      <div class="text-4xl font-extrabold text-gray-900 mb-8 tracking-tight">
        {{ formatCurrency(account.balance, account.currency) }}
      </div>

      <!-- Action Buttons -->
      <div class="flex gap-4 w-full max-w-sm">
        <button 
          id="cta-transfer"
          @click="goToTransfers"
          class="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-bold py-4 rounded-2xl shadow-lg shadow-blue-200 transition-all active:scale-95 flex flex-col items-center justify-center gap-2"
        >
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4"></path></svg>
          <span>Transfer</span>
        </button>
        
        <!-- Placeholder for potential future actions like 'Add Money' or 'Details' -->
        <button class="flex-1 bg-white hover:bg-gray-50 text-gray-700 font-bold py-4 rounded-2xl shadow-sm border border-gray-200 transition-all flex flex-col items-center justify-center gap-2">
           <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
           <span>Details</span>
        </button>
      </div>

      <!-- Transactions Placeholder (Visual only) -->
      <div class="w-full max-w-md mt-10">
        <h3 class="font-bold text-gray-900 mb-4">Recent Transactions</h3>
        <div class="space-y-4">
          <div class="flex items-center justify-between p-3 bg-white rounded-xl shadow-sm">
            <div class="flex items-center gap-3">
              <div class="w-10 h-10 bg-gray-100 rounded-full flex items-center justify-center">🛍️</div>
              <div>
                <div class="font-bold text-sm">Amazon</div>
                <div class="text-xs text-gray-500">Yesterday</div>
              </div>
            </div>
            <span class="font-bold text-gray-900">-$24.99</span>
          </div>
           <div class="flex items-center justify-between p-3 bg-white rounded-xl shadow-sm">
            <div class="flex items-center gap-3">
              <div class="w-10 h-10 bg-gray-100 rounded-full flex items-center justify-center">☕</div>
              <div>
                <div class="font-bold text-sm">Starbucks</div>
                <div class="text-xs text-gray-500">Today</div>
              </div>
            </div>
            <span class="font-bold text-gray-900">-$5.40</span>
          </div>
        </div>
      </div>

    </div>
    
    <div v-else class="flex-1 flex items-center justify-center text-gray-500">
      Account not found.
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'ACCOUNT_DETAIL',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const account = computed(() => {
      const id = signatureStore.accounts_selected_account_id
      return dataStore.accounts.find(a => a.id === id)
    })

    const formatCurrency = (value, currency) => {
      if (!currency) return value
      return new Intl.NumberFormat('en-US', { style: 'currency', currency: currency }).format(value)
    }

    const goBack = () => {
      signatureStore.setCurrentPageId('ACCOUNTS_DASHBOARD')
      router.push({ name: 'ACCOUNTS_DASHBOARD' })
    }

    const goToTransfers = () => {
      signatureStore.setCurrentPageId('PAYMENTS_LIST')
      router.push({ name: 'PAYMENTS_LIST' })
    }

    return {
      account,
      formatCurrency,
      goBack,
      goToTransfers
    }
  }
}
</script>