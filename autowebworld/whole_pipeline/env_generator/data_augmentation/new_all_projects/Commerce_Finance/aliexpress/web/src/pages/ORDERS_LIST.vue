<template>
  <div class="min-h-screen bg-gray-50 pb-20 font-sans">
    <header class="bg-white shadow-sm px-4 py-3 sticky top-0 z-20">
       <div class="flex items-center space-x-3 mb-3">
          <button 
            id="orders-back-account" 
            class="p-2 -ml-2 hover:bg-gray-100 rounded-full transition-colors"
            @click="handleBackAccount"
          >
             <svg class="w-6 h-6 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
          </button>
          <div class="flex-1 relative">
             <input 
               id="orders-search-input"
               type="text" 
               placeholder="Search orders" 
               class="w-full bg-gray-100 border-none rounded-full py-2 pl-10 pr-4 text-sm focus:ring-2 focus:ring-red-500 focus:bg-white transition-all"
               v-model="searchQuery"
               @keyup.enter="handleSearch"
             />
             <svg class="w-5 h-5 text-gray-400 absolute left-3 top-2.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
          </div>
       </div>
       
       <!-- Filter Tabs -->
       <div class="flex items-center space-x-4 overflow-x-auto no-scrollbar border-b border-gray-100 -mx-4 px-4">
          <div class="pb-2 border-b-2 border-red-600 text-red-600 font-bold text-sm whitespace-nowrap cursor-pointer">All</div>
          <div 
             id="filter-orders-pending-checkbox"
             class="pb-2 border-b-2 text-sm whitespace-nowrap cursor-pointer transition-colors hover:text-red-600"
             :class="signatureStore.ORDERS_LIST_filters_applied ? 'border-red-600 text-red-600 font-bold' : 'border-transparent text-gray-500 font-medium'"
             @click="handleFilterPending"
          >
             Unpaid
          </div>
          <div class="pb-2 border-b-2 border-transparent text-gray-500 font-medium text-sm whitespace-nowrap cursor-pointer hover:text-red-600">To Ship</div>
          <div class="pb-2 border-b-2 border-transparent text-gray-500 font-medium text-sm whitespace-nowrap cursor-pointer hover:text-red-600">Shipped</div>
       </div>
    </header>

    <div id="orders-list-container" class="p-4 space-y-4">
       <div 
         v-for="i in 4" 
         :key="i"
         :class="[
           'bg-white rounded-xl p-4 shadow-sm cursor-pointer hover:shadow-md transition-shadow',
           `data-id-ord-${i}`,
           signatureStore.matched_item_id === `ord-${i}` ? 'order-row-matched' : 'order-row-visible'
         ]"
         @click="handleOpenOrder(`ord-${i}`)"
       >
          <div class="flex justify-between items-center mb-3 border-b border-gray-50 pb-2">
             <div class="flex items-center space-x-2">
                <span class="text-xs font-bold text-gray-900">Store Name</span>
                <svg class="w-3 h-3 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path></svg>
             </div>
             <span class="text-xs text-red-600 font-bold uppercase">Pending Payment</span>
          </div>
          <div class="flex space-x-3">
             <div class="w-20 h-20 bg-gray-100 rounded-lg flex-shrink-0"></div>
             <div class="flex-1 min-w-0">
                <h3 class="text-sm font-medium text-gray-900 line-clamp-2">Product Name Placeholder That Is Quite Long</h3>
                <p class="text-xs text-gray-500 mt-1">Variant: Black</p>
                <div class="flex justify-end mt-2">
                   <span class="text-xs text-gray-500 mr-1">Total:</span>
                   <span class="text-sm font-bold text-gray-900">$24.99</span>
                </div>
             </div>
          </div>
          <div class="flex justify-end space-x-2 mt-4 pt-2 border-t border-gray-50">
             <button class="px-3 py-1.5 border border-gray-300 rounded-full text-xs font-medium text-gray-600 hover:bg-gray-50">Cancel</button>
             <button class="px-3 py-1.5 bg-red-600 text-white rounded-full text-xs font-bold hover:bg-red-700 shadow-sm">Pay Now</button>
          </div>
       </div>
    </div>
  </div>
</template>

<script>
import { ref, watch, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'ORDERS_LIST',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const searchQuery = ref('')

    const handleBackAccount = async () => {
       signatureStore.currentPageId = 'ACCOUNT_OVERVIEW'
       await router.push({ name: 'ACCOUNT_OVERVIEW' })
    }

    const handleSearch = () => {
       signatureStore.ORDERS_LIST_has_searched = true
       // Mock match
       signatureStore.matched_item_id = 'ord-1'
    }

    const handleFilterPending = () => {
       signatureStore.ORDERS_LIST_filters_applied = true
    }

    const handleOpenOrder = async (id) => {
       if(signatureStore.ORDERS_LIST_has_searched) signatureStore.ORDERS_LIST_has_searched = null
       if(signatureStore.ORDERS_LIST_viewport_anchor_id) signatureStore.ORDERS_LIST_viewport_anchor_id = null
       
       signatureStore.selected_item_id = id
       signatureStore.currentPageId = 'ORDER_DETAIL'
       await router.push({ name: 'ORDER_DETAIL' })
    }

    // Scroll handler
    watch(() => signatureStore.ORDERS_LIST_viewport_anchor_id, async (newId) => {
      if (newId) {
        await nextTick()
        const element = document.querySelector(`.data-id-${newId}`)
        if (element) {
          element.scrollIntoView({ behavior: 'smooth' })
        }
      }
    })

    return {
       signatureStore,
       searchQuery,
       handleBackAccount,
       handleSearch,
       handleFilterPending,
       handleOpenOrder
    }
  }
}
</script>