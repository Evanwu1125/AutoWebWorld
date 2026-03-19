<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <header class="bg-white shadow-sm z-10">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
        <h1 class="text-2xl font-bold text-[#005DAA]">Billing</h1>
        <button id="back-dashboard" @click="handleBack" class="text-gray-600 hover:text-gray-900">
          Back
        </button>
      </div>
    </header>

    <main class="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
      <!-- Search -->
      <div class="bg-white p-4 rounded-lg shadow mb-6">
        <div class="relative">
           <input
             id="billing-search-input"
             type="text"
             placeholder="Search bills..."
             v-model="searchQuery"
             @keyup.enter="handleSearch"
             class="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:ring-[#009CDE] focus:border-[#009CDE]"
           />
           <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
             <svg class="h-5 w-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
           </div>
        </div>
      </div>

      <!-- List -->
      <div id="billing-list-container" class="space-y-4">
        <div 
          v-for="bill in dataStore.bills" 
          :key="bill.id"
          class="bg-white rounded-lg shadow hover:shadow-md transition-shadow duration-200"
          :class="{
             'ring-2 ring-green-500': bill.id === matchedId,
             'ring-2 ring-blue-500': bill.id === store.billing_list_viewport_anchor_id
          }"
        >
          <div 
             :id="bill.id === matchedId ? 'billing-list-item-matched' : 'billing-list-item-visible'"
             :class="`data-id-${bill.id} p-6 flex items-center space-x-4 cursor-pointer`"
             @click="handleSelectBill(bill)"
          >
             <div class="h-12 w-12 bg-gray-100 rounded-full flex items-center justify-center text-gray-500">
               <span class="font-bold text-lg">$</span>
             </div>
             
             <div class="flex-1 min-w-0">
               <div class="flex justify-between">
                 <h3 class="text-lg font-bold text-gray-900">{{ bill.description }}</h3>
                 <span 
                   class="px-2 py-1 text-xs font-semibold rounded-full"
                   :class="bill.status === 'Paid' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'"
                 >
                   {{ bill.status }}
                 </span>
               </div>
               <div class="flex justify-between mt-1">
                  <p class="text-sm text-gray-500">{{ bill.date }}</p>
                  <p class="text-lg font-bold text-gray-900">${{ bill.amount.toFixed(2) }}</p>
               </div>
             </div>
             
             <div class="self-center">
                <svg class="h-6 w-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path></svg>
             </div>
          </div>
        </div>

        <div v-if="dataStore.bills.length === 0" class="text-center py-12">
           <p class="text-gray-500">No bills found.</p>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'BILLING_OVERVIEW',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const searchQuery = ref('')
    const matchedId = ref(null)

    const handleSearch = () => {
      // ACT_BILL_SEARCH
      const match = dataStore.bills.find(b => b.description.toLowerCase().includes(searchQuery.value.toLowerCase()))
      if (match) {
        store.matched_bill_id = match.id
        store.billing_list_has_searched = true
        matchedId.value = match.id
      } else {
        matchedId.value = null
      }
    }

    const handleSelectBill = async (bill) => {
      // ACT_BILL_OPEN_MATCHED, ACT_BILL_OPEN_ANY
      store.selected_bill_id = bill.id
      
      store.billing_list_has_searched = false
      store.billing_list_viewport_anchor_id = null

      store.setCurrentPageId('BILL_DETAIL')
      await router.push({ name: 'BILL_DETAIL' })
    }

    const handleBack = async () => {
      // ACT_BILL_BACK_DASH
      store.setCurrentPageId('DASHBOARD')
      await router.push({ name: 'DASHBOARD' })
    }

    return {
      store,
      dataStore,
      searchQuery,
      matchedId,
      handleSearch,
      handleSelectBill,
      handleBack
    }
  }
}
</script>