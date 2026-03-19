<template>
  <div class="success-page min-h-screen bg-gray-50 flex items-center justify-center p-4">
    <div class="bg-white max-w-md w-full rounded-2xl shadow-xl overflow-hidden text-center p-8 animate-fade-in-up">
      <div class="w-20 h-20 bg-yellow-100 rounded-full flex items-center justify-center mx-auto mb-6">
        <svg class="w-10 h-10 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
      </div>
      
      <h1 class="text-3xl font-bold text-gray-900 mb-2">Quick Order Placed!</h1>
      <p class="text-gray-600 mb-8">Your Buy Now order is confirmed.</p>

      <div class="space-y-3">
        <button 
          id="buy-now-success-view-orders"
          @click="handleViewOrders"
          class="w-full bg-[#0071DC] text-white font-bold py-3 rounded-full hover:bg-[#005bb5] shadow-md transition-colors"
        >
          View Details
        </button>
        <button 
          id="buy-now-success-go-home"
          @click="handleGoHome"
          class="w-full bg-white text-gray-700 font-bold py-3 rounded-full border border-gray-300 hover:bg-gray-50 transition-colors"
        >
          Done
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'ORDER_PLACED_BUY_NOW_SUCCESS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const handleViewOrders = async () => {
      // FSM: ACT_BUY_NOW_SUCCESS_VIEW_ORDERS
      store.currentPageId = 'ORDER_HISTORY'
      await router.push({ name: 'ORDER_HISTORY' })
    }

    const handleGoHome = async () => {
      // FSM: ACT_BUY_NOW_SUCCESS_GO_HOME
      store.currentPageId = 'HOME'
      await router.push({ name: 'HOME' })
    }

    return {
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