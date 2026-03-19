<template>
  <div class="min-h-screen bg-gray-50 text-gray-900 font-sans">
    <nav class="bg-white border-b border-gray-200">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <span class="text-xl font-bold text-[#008060]">Order History</span>
            <span 
                id="orders-back-account" 
                @click="goBackAccount"
                class="text-gray-500 hover:text-[#008060] cursor-pointer text-sm font-medium"
            >
                Back to Account
            </span>
        </div>
    </nav>

    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div class="flex flex-col lg:flex-row gap-8">
        <!-- Sidebar Filters -->
        <aside class="w-full lg:w-64 flex-shrink-0 space-y-8">
             <div class="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                 <input 
                    id="orders-search-input"
                    type="text" 
                    v-model="searchQuery"
                    @keypress.enter="performSearch"
                    placeholder="Search Order ID..."
                    class="w-full border-gray-300 rounded-lg shadow-sm focus:border-[#008060] focus:ring focus:ring-[#008060] py-2 px-4 mb-4"
                 />
            </div>

            <div class="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                <h3 class="text-lg font-semibold mb-4">Sort By</h3>
                <div class="relative" id="orders-sort-dropdown">
                    <select
                        v-model="selectedSort"
                        @change="handleSort"
                        class="w-full border-gray-300 rounded-lg shadow-sm focus:border-[#008060] focus:ring focus:ring-[#008060] py-2 px-3"
                    >
                        <option value="">Default</option>
                        <option value="newest" id="orders-sort-option-newest">Newest</option>
                        <option value="oldest" id="orders-sort-option-oldest">Oldest</option>
                    </select>
                </div>
            </div>

            <div class="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                <h3 class="text-lg font-semibold mb-4">Date Range</h3>
                <input 
                    id="orders-date-slider"
                    type="range" 
                    v-model.number="dateFilter" 
                    :min="0" 
                    :max="100" 
                    step="1"
                    @change="handleSliderChange"
                    class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-[#008060]"
                />
                 <div class="text-sm text-gray-500 mt-2 text-center">Past {{ dateFilter }} days</div>
            </div>

             <div class="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                <label class="flex items-center space-x-3 cursor-pointer">
                    <input 
                        id="orders-filter-open-checkbox"
                        type="checkbox" 
                        v-model="openFilter" 
                        @change="handleCheckboxChange"
                        class="form-checkbox h-5 w-5 text-[#008060] rounded focus:ring-[#008060] border-gray-300"
                    />
                    <span class="text-gray-700">Open Orders Only</span>
                </label>
            </div>
        </aside>

        <!-- Order List -->
        <div class="flex-1" id="order-list">
            <div v-if="filteredOrders.length === 0" class="bg-white p-12 text-center rounded-xl border border-gray-200">
                No orders found.
            </div>
            
            <div class="space-y-4" v-else>
                <div 
                    v-for="order in filteredOrders" 
                    :key="order.id"
                    :class="[
                        'bg-white p-6 rounded-xl shadow-sm border border-gray-100 hover:shadow-md transition-shadow cursor-pointer flex justify-between items-center',
                        `data-id-${order.id}`,
                        isFiltered ? 'order-row-filtered' : '',
                        isSearched ? 'order-row-matched' : '',
                        !isFiltered && !isSearched ? 'order-row-visible' : ''
                    ]"
                    @click="openOrder(order.id)"
                >
                    <div>
                        <div class="font-bold text-lg text-gray-900">{{ order.order_number }}</div>
                        <div class="text-sm text-gray-500">{{ order.date }}</div>
                    </div>
                    <div class="text-right">
                        <div class="font-bold text-gray-900">${{ order.total.toFixed(2) }}</div>
                        <div 
                            :class="[
                                'text-xs font-bold px-2 py-1 rounded inline-block mt-1',
                                order.fulfillment_status === 'Fulfilled' ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'
                            ]"
                        >
                            {{ order.fulfillment_status }}
                        </div>
                    </div>
                </div>
            </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { computed, ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'ORDER_HISTORY',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const searchQuery = ref('')
    const selectedSort = ref('')
    const dateFilter = ref(100)
    const openFilter = ref(false)

    const isFiltered = computed(() => signatureStore.orders_filters_applied === true)
    const isSearched = computed(() => signatureStore.orders_list_has_searched === true)

    onMounted(() => {
        signatureStore.resetTransientState()
        selectedSort.value = 'newest'
        dateFilter.value = 100
        openFilter.value = false
    })

    const filteredOrders = computed(() => {
        let items = [...dataStore.orders]

        if (searchQuery.value) {
            items = items.filter(o => o.order_number.includes(searchQuery.value) || o.id.includes(searchQuery.value))
        }

        if (openFilter.value) {
            items = items.filter(o => o.fulfillment_status === 'Unfulfilled')
        }

        // Mock date filter logic (assume current date is late 2023 per mock data)
        // Just return all for mock simplicity unless strictly needed logic

        if (selectedSort.value) {
            if (selectedSort.value === 'newest') {
                items.sort((a, b) => new Date(b.date) - new Date(a.date))
            } else if (selectedSort.value === 'oldest') {
                items.sort((a, b) => new Date(a.date) - new Date(b.date))
            }
        }

        return items
    })

    const performSearch = () => {
        if (searchQuery.value) {
            signatureStore.orders_list_has_searched = true
            signatureStore.orders_matched_order_id = 'MATCHED_ANY'
        }
    }

    const handleSort = () => {
        signatureStore.orders_filters_applied = true
    }

    const handleSliderChange = () => {
        signatureStore.orders_filters_applied = true
    }

    const handleCheckboxChange = () => {
        signatureStore.orders_filters_applied = true
    }

    const openOrder = async (orderId) => {
        if (isFiltered.value) {
            signatureStore.orders_selected_order_id = orderId
            signatureStore.orders_filters_applied = null
        } else if (isSearched.value) {
             signatureStore.orders_selected_order_id = orderId
             signatureStore.orders_list_has_searched = null
        } else {
             signatureStore.orders_viewport_anchor_id = orderId
             signatureStore.orders_selected_order_id = orderId
             signatureStore.orders_viewport_anchor_id = null
        }
        
        signatureStore.currentPageId = 'ORDER_DETAIL'
        await router.push({ name: 'ORDER_DETAIL', params: { id: orderId } })
    }

    const goBackAccount = async () => {
        signatureStore.currentPageId = 'ACCOUNT_DASHBOARD'
        await router.push({ name: 'ACCOUNT_DASHBOARD' })
    }

    return {
        searchQuery,
        selectedSort,
        dateFilter,
        openFilter,
        filteredOrders,
        isFiltered,
        isSearched,
        performSearch,
        handleSort,
        handleSliderChange,
        handleCheckboxChange,
        openOrder,
        goBackAccount
    }
  }
}
</script>