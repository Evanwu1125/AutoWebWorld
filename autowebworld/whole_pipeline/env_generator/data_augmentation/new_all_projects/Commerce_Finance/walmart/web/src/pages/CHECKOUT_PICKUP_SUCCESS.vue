<template>
  <div class="success-page min-h-screen bg-gray-50 flex items-center justify-center p-4">
    <div class="bg-white max-w-md w-full rounded-2xl shadow-xl overflow-hidden text-center p-8 animate-fade-in-up">
      <div class="w-20 h-20 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-6">
        <svg class="w-10 h-10 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" /></svg>
      </div>
      
      <h1 class="text-3xl font-bold text-gray-900 mb-2">Ready for Pickup!</h1>
      <p class="text-gray-600 mb-6">Your order #{{ store.order_id }} has been placed.</p>
      
      <div class="bg-gray-50 rounded-xl p-4 mb-8 text-left border border-gray-100">
        <h3 class="font-bold text-sm text-gray-500 uppercase mb-2">Pickup Details</h3>
        <p class="font-medium">Available today after 2pm</p>
        <p class="text-sm text-gray-500 mt-1">Bring your ID and confirmation email.</p>
      </div>

      <div class="space-y-3">
        <button 
          id="pickup-success-view-orders"
          @click="handleViewOrders"
          class="w-full bg-[#0071DC] text-white font-bold py-3 rounded-full hover:bg-[#005bb5] shadow-md transition-colors"
        >
          View My Orders
        </button>
        <button 
          id="pickup-success-go-home"
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
  name: 'CHECKOUT_PICKUP_SUCCESS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const handleViewOrders = async () => {
      // FSM: ACT_PICKUP_SUCCESS_VIEW_ORDERS
      store.currentPageId = 'ORDER_HISTORY'
      await router.push({ name: 'ORDER_HISTORY' })
    }

    const handleGoHome = async () => {
      // FSM: ACT_PICKUP_SUCCESS_GO_HOME
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