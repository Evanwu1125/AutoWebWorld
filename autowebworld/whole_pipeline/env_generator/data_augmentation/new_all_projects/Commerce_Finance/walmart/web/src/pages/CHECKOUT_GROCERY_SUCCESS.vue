<template>
  <div class="success-page min-h-screen bg-gray-50 flex items-center justify-center p-4">
    <div class="bg-white max-w-md w-full rounded-2xl shadow-xl overflow-hidden text-center p-8 animate-fade-in-up">
      <div class="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-6">
        <svg class="w-10 h-10 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7" /></svg>
      </div>
      
      <h1 class="text-3xl font-bold text-gray-900 mb-2">Order Confirmed!</h1>
      <p class="text-gray-600 mb-6">Your grocery order #{{ store.order_id }} has been placed.</p>
      
      <div class="bg-green-50 rounded-xl p-4 mb-8 text-left border border-green-100">
        <h3 class="font-bold text-sm text-green-700 uppercase mb-2">Delivery Time</h3>
        <p class="font-medium text-gray-900">{{ store.grocery_delivery_slot }}</p>
      </div>

      <div class="space-y-3">
        <button 
          id="grocery-success-view-orders"
          @click="handleViewOrders"
          class="w-full bg-[#2A8703] text-white font-bold py-3 rounded-full hover:bg-[#237002] shadow-md transition-colors"
        >
          View My Orders
        </button>
        <button 
          id="grocery-success-go-home"
          @click="handleGoHome"
          class="w-full bg-white text-gray-700 font-bold py-3 rounded-full border border-gray-300 hover:bg-gray-50 transition-colors"
        >
          Back to Home
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'CHECKOUT_GROCERY_SUCCESS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const handleViewOrders = async () => {
      // FSM: ACT_GROCERY_SUCCESS_VIEW_ORDERS
      store.currentPageId = 'ORDER_HISTORY'
      await router.push({ name: 'ORDER_HISTORY' })
    }

    const handleGoHome = async () => {
      // FSM: ACT_GROCERY_SUCCESS_GO_HOME
      store.currentPageId = 'HOME'
      await router.push({ name: 'HOME' })
    }

    return {
      store,
      handleViewOrders,
      handleGoHome
    }
  }
}
</script>
<style scoped>
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}
.animate-fade-in-up {
  animation: fadeInUp 0.5s ease-out forwards;
}
</style>