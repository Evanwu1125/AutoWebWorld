<template>
  <div class="order-history-page min-h-screen bg-gray-50 flex flex-col">
    <header class="bg-[#0071DC] text-white p-4 sticky top-0 z-30 shadow-md">
      <div class="max-w-7xl mx-auto flex items-center justify-between">
         <div id="order-history-logo-home" @click="handleGoHome" class="font-bold text-xl cursor-pointer flex items-center gap-2">
            <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" /></svg>
            Walmart
         </div>
         <h1 class="text-lg font-medium">Order History</h1>
      </div>
    </header>

    <main class="flex-1 max-w-5xl mx-auto w-full p-4 md:p-8">
      <div class="flex flex-col md:flex-row gap-4 mb-8">
        <!-- Search -->
        <div class="relative flex-1">
           <input 
             id="order-search-input"
             type="text" 
             v-model="searchQuery"
             @keydown.enter="handleSearch"
             placeholder="Search your orders" 
             class="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-full focus:ring-2 focus:ring-blue-500 outline-none"
           />
           <svg class="w-5 h-5 text-gray-400 absolute left-3 top-1/2 -translate-y-1/2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" /></svg>
        </div>

        <!-- Filters & Sort -->
        <div class="flex gap-4">
           <!-- Status Filter -->
           <label class="flex items-center gap-2 bg-white px-4 py-2 rounded-full border shadow-sm cursor-pointer hover:bg-gray-50">
             <input 
               id="filter-order-status-checkbox"
               type="checkbox" 
               v-model="statusFilter"
               @change="handleFilterStatus"
               class="rounded text-blue-600 focus:ring-blue-500"
             />
             <span class="text-sm font-medium">Delivered Only</span>
           </label>

           <!-- Sort Dropdown -->
           <div class="relative">
             <button 
               id="order-sort-dropdown"
               @click="showSort = !showSort"
               class="flex items-center gap-2 bg-white px-4 py-2 rounded-full border shadow-sm text-sm font-medium hover:bg-gray-50"
             >
               Sort: {{ currentSortLabel }}
               <svg class="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" /></svg>
             </button>
             
             <div v-if="showSort" class="absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-xl py-1 border border-gray-100 z-20">
               <div 
                 id="order-sort-option-newest" 
                 @click="handleSort('newest')"
                 class="px-4 py-2 text-sm hover:bg-gray-100 cursor-pointer"
               >
                 Newest to Oldest
               </div>
               <div 
                 id="order-sort-option-oldest" 
                 @click="handleSort('oldest')"
                 class="px-4 py-2 text-sm hover:bg-gray-100 cursor-pointer"
               >
                 Oldest to Newest
               </div>
             </div>
           </div>
        </div>
      </div>

      <!-- Orders List -->
      <div id="order-list" class="space-y-4">
        <div 
          v-for="order in filteredOrders" 
          :key="order.id"
          :class="[
            'bg-white rounded-xl shadow-sm border border-gray-100 p-6 cursor-pointer hover:shadow-md transition-shadow group',
            getOrderClass(order)
          ]"
          :data-id="order.id"
          @click="handleOrderClick(order)"
        >
          <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-4 pb-4 border-b">
            <div>
              <div class="text-sm text-gray-500 mb-1">Order Placed: {{ order.date }}</div>
              <div class="font-bold text-gray-900">Order #{{ order.id }}</div>
            </div>
            <div class="mt-2 sm:mt-0 text-right">
              <div class="text-sm text-gray-500 mb-1">Total</div>
              <div class="font-bold text-gray-900">${{ order.total.toFixed(2) }}</div>
            </div>
          </div>
          
          <div class="flex items-center justify-between">
            <div class="flex items-center gap-2">
              <span 
                class="inline-block w-3 h-3 rounded-full"
                :class="{
                  'bg-green-500': order.status === 'delivered',
                  'bg-blue-500': order.status === 'shipped',
                  'bg-yellow-500': order.status === 'processing',
                  'bg-red-500': order.status === 'cancelled'
                }"
              ></span>
              <span class="font-medium capitalize text-gray-700">{{ order.status }}</span>
            </div>
            <div class="text-[#0071DC] font-medium group-hover:underline">View Details &rarr;</div>
          </div>
        </div>

        <div v-if="filteredOrders.length === 0" class="text-center py-20 bg-white rounded-xl shadow-sm">
           <div class="text-6xl mb-4">📦</div>
           <h3 class="text-xl font-bold text-gray-900">No orders found</h3>
           <p class="text-gray-500 mt-2">Try adjusting filters or search.</p>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'ORDER_HISTORY',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const searchQuery = ref('')
    const statusFilter = ref(false)
    const currentSort = ref(null)
    const showSort = ref(false)

    const orders = computed(() => dataStore.orders)

    const filteredOrders = computed(() => {
      let res = [...orders.value]

      if (searchQuery.value) {
        const q = searchQuery.value.toLowerCase()
        res = res.filter(o => o.id.toLowerCase().includes(q) || o.status.toLowerCase().includes(q))
      }

      if (statusFilter.value) {
        res = res.filter(o => o.status === 'delivered')
      }

      if (currentSort.value === 'newest') {
        res.sort((a, b) => new Date(b.date) - new Date(a.date))
      } else {
        res.sort((a, b) => new Date(a.date) - new Date(b.date))
      }

      return res
    })

    const currentSortLabel = computed(() => {
      if (!currentSort.value) return 'Default'
      return currentSort.value === 'newest' ? 'Newest' : 'Oldest'
    })

    const getOrderClass = (order) => {
      // FSM Logic for selectors: 
      // ACT_ORDER_OPEN_FILTERED_ORDER -> .order-card-filtered
      // ACT_ORDER_OPEN_MATCHED_ORDER -> .order-card-matched
      // ACT_ORDER_OPEN_ANY_ORDER -> .order-card-visible
      
      const classes = ['order-card-visible'] // Default
      
      if (statusFilter.value || currentSort.value !== 'newest') { // Assuming 'newest' is default in logic or implicit
         // Actually FSM sets filters_applied when ANY filter/sort action happens.
         // We can use store state for stricter mapping, but computed is safer for visual correctness.
         if (store.order_history_filters_applied) classes.push('order-card-filtered')
      }

      if (searchQuery.value && store.order_history_has_searched) {
        classes.push('order-card-matched')
      }

      return classes.join(' ')
    }

    // Handlers
    const handleSearch = () => {
      // FSM: ACT_ORDER_SEARCH
      store.order_history_has_searched = true
      if (filteredOrders.value.length > 0) {
        store.matched_order_id = filteredOrders.value[0].id
      }
    }

    const handleFilterStatus = () => {
      // FSM: ACT_ORDER_FILTER_STATUS_CHECKBOX
      store.order_history_filters_applied = true
    }

    const handleSort = (val) => {
      // FSM: ACT_ORDER_FILTER_SORT
      currentSort.value = val
      showSort.value = false
      store.order_history_filters_applied = true
    }

    const handleOrderClick = async (order) => {
      store.selected_order_id = order.id
      
      // Clear flags
      store.order_history_filters_applied = null
      store.order_history_has_searched = null
      store.order_history_viewport_anchor_id = null

      store.currentPageId = 'ORDER_DETAIL'
      await router.push({ name: 'ORDER_DETAIL', params: { id: order.id } })
    }

    const handleGoHome = async () => {
      // FSM: ACT_ORDER_HISTORY_BACK_TO_HOME
      store.currentPageId = 'HOME'
      await router.push({ name: 'HOME' })
    }

    return {
      searchQuery,
      statusFilter,
      filteredOrders,
      showSort,
      currentSortLabel,
      handleSearch,
      handleFilterStatus,
      handleSort,
      handleOrderClick,
      handleGoHome,
      getOrderClass
    }
  }
}
</script>