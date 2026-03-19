<template>
  <div class="min-h-screen bg-gray-50 text-gray-900 font-sans">
     <nav class="bg-white border-b border-gray-200">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <span 
                id="order-back-to-orders" 
                @click="goBackOrders"
                class="text-gray-500 hover:text-[#008060] cursor-pointer text-sm font-medium flex items-center"
            >
                <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
                </svg>
                Back to Orders
            </span>
            <span class="font-bold text-gray-900" v-if="order">{{ order.order_number }}</span>
        </div>
    </nav>

    <main class="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-12" v-if="order">
        <div class="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden mb-6">
            <div class="p-6 border-b border-gray-100 flex justify-between items-center">
                <h1 class="text-xl font-bold">Order Details</h1>
                <span class="bg-green-100 text-green-800 text-xs font-bold px-2 py-1 rounded">{{ order.fulfillment_status }}</span>
            </div>
            <div class="p-6 space-y-4">
                <div v-for="(item, idx) in order.items" :key="idx" class="flex justify-between items-center">
                    <div>
                        <div class="font-medium text-gray-900">{{ item.title }}</div>
                        <div class="text-sm text-gray-500">Qty: {{ item.quantity }}</div>
                    </div>
                    <div class="font-medium text-gray-900">${{ item.price.toFixed(2) }}</div>
                </div>
            </div>
            <div class="bg-gray-50 p-6 border-t border-gray-100 flex justify-between items-center">
                 <span class="font-bold text-gray-900">Total</span>
                 <span class="font-bold text-xl text-gray-900">${{ order.total.toFixed(2) }}</span>
            </div>
        </div>
        
        <div class="text-right">
             <button 
                id="order-view-summary" 
                @click="viewSummary"
                class="bg-[#008060] hover:bg-[#004C3F] text-white font-bold py-3 px-6 rounded-lg shadow-md transition-colors"
            >
                View Full Summary
            </button>
        </div>
    </main>
    
    <div v-else class="text-center py-20">Loading order...</div>
  </div>
</template>

<script>
import { computed, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'ORDER_DETAIL',
  setup() {
    const router = useRouter()
    const route = useRoute()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const order = computed(() => dataStore.orders.find(o => o.id === route.params.id))

    onMounted(() => {
        if (order.value) {
            signatureStore.selected_order_id = order.value.id
        }
    })

    const goBackOrders = async () => {
        signatureStore.currentPageId = 'ORDER_HISTORY'
        await router.push({ name: 'ORDER_HISTORY' })
    }

    const viewSummary = async () => {
        signatureStore.currentPageId = 'ORDER_SUMMARY'
        await router.push({ name: 'ORDER_SUMMARY', params: { id: route.params.id } })
    }

    return {
        order,
        goBackOrders,
        viewSummary
    }
  }
}
</script>