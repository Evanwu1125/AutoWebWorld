<template>
  <div class="order-detail-page min-h-screen bg-gray-50 flex flex-col">
    <header class="bg-[#0071DC] text-white p-4 sticky top-0 z-30 shadow-md">
      <div class="max-w-7xl mx-auto flex items-center gap-4">
        <div 
          id="order-detail-back-to-history" 
          @click="handleBackToHistory"
          class="cursor-pointer p-2 hover:bg-white/10 rounded-full transition-colors flex items-center gap-1"
        >
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" /></svg>
          <span class="text-sm font-medium">Order History</span>
        </div>
        <h1 class="text-lg font-bold">Order Details</h1>
      </div>
    </header>

    <main v-if="order" class="flex-1 max-w-4xl mx-auto w-full p-4 md:p-8">
      <div class="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden mb-6">
        <div class="p-6 border-b bg-gray-50 flex flex-col md:flex-row justify-between md:items-center gap-4">
          <div>
            <h2 class="text-xl font-bold text-gray-900">Order #{{ order.id }}</h2>
            <div class="text-gray-500 text-sm">Placed on {{ order.date }}</div>
          </div>
          <div class="flex items-center gap-2">
            <span class="font-medium">Status:</span>
            <span 
              class="px-3 py-1 rounded-full text-sm font-bold uppercase tracking-wide"
              :class="{
                  'bg-green-100 text-green-700': order.status === 'delivered',
                  'bg-blue-100 text-blue-700': order.status === 'shipped',
                  'bg-yellow-100 text-yellow-700': order.status === 'processing',
                  'bg-red-100 text-red-700': order.status === 'cancelled'
              }"
            >
              {{ order.status }}
            </span>
          </div>
        </div>

        <div class="p-6">
           <h3 class="font-bold text-lg mb-4">Items Ordered</h3>
           <div class="space-y-4">
             <div v-for="item in orderItems" :key="item.id" class="flex gap-4 py-4 border-b last:border-0">
                <div class="w-20 h-20 bg-gray-50 rounded-lg p-2 border border-gray-100 flex-shrink-0">
                   <img :src="item.image" :alt="item.name" class="w-full h-full object-contain mix-blend-multiply" />
                </div>
                <div class="flex-1">
                   <div class="font-medium text-gray-900 mb-1">{{ item.name }}</div>
                   <div class="text-gray-500 text-sm">Qty: 1</div>
                </div>
                <div class="font-bold text-gray-900">${{ item.price.toFixed(2) }}</div>
             </div>
           </div>
        </div>

        <div class="p-6 bg-gray-50 border-t flex flex-col md:flex-row justify-between md:items-center gap-4">
           <div>
             <div class="text-sm text-gray-500 mb-1">Total Amount</div>
             <div class="text-2xl font-bold text-gray-900">${{ order.total.toFixed(2) }}</div>
           </div>
           
           <button 
             id="order-detail-reorder-button"
             @click="handleReorder"
             class="bg-[#0071DC] text-white font-bold py-3 px-8 rounded-full shadow-md hover:bg-[#005bb5] transition-all transform hover:-translate-y-0.5"
           >
             Reorder Items
           </button>
        </div>
      </div>
    </main>
    
    <div v-else class="flex-1 flex items-center justify-center">
       <div class="animate-spin rounded-full h-12 w-12 border-4 border-blue-200 border-t-blue-600"></div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'ORDER_DETAIL',
  setup() {
    const route = useRoute()
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const orderId = route.params.id || store.selected_order_id
    const order = computed(() => dataStore.orders.find(o => o.id === orderId))
    
    const orderItems = computed(() => {
      if (!order.value) return []
      return order.value.items.map(itemId => {
        // Search both collections
        let item = dataStore.electronics.find(p => p.id === itemId)
        if (!item) item = dataStore.groceries.find(p => p.id === itemId)
        return item || { id: itemId, name: 'Unknown Item', price: 0, image: '' }
      })
    })

    const handleReorder = async () => {
      // FSM: ACT_ORDER_DETAIL_REORDER
      // Add items to cart
      if (order.value) {
        order.value.items.forEach(itemId => {
           store.cart_items.push({ id: itemId, qty: 1 })
        })
      }
      
      store.currentPageId = 'CART'
      await router.push({ name: 'CART' })
    }

    const handleBackToHistory = async () => {
      // FSM: ACT_ORDER_DETAIL_BACK_TO_HISTORY
      store.currentPageId = 'ORDER_HISTORY'
      await router.push({ name: 'ORDER_HISTORY' })
    }

    return {
      order,
      orderItems,
      handleReorder,
      handleBackToHistory
    }
  }
}
</script>