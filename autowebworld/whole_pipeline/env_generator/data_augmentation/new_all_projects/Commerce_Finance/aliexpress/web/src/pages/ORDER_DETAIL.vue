<template>
  <div class="min-h-screen bg-gray-50 font-sans pb-24">
    <header class="bg-white shadow-sm px-4 py-3 flex items-center sticky top-0 z-20">
      <button 
        id="order-back-orders" 
        class="p-2 -ml-2 hover:bg-gray-100 rounded-full transition-colors"
        @click="handleBackOrders"
      >
        <svg class="w-6 h-6 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
      </button>
      <h1 class="text-lg font-bold text-gray-900 ml-2">Order Details</h1>
    </header>

    <div class="bg-red-600 text-white p-6 shadow-md">
       <h2 class="text-xl font-bold">Awaiting Payment</h2>
       <p class="text-sm opacity-90 mt-1">Please pay within 23 hours 59 minutes</p>
    </div>

    <div class="p-4 space-y-4">
       <!-- Address -->
       <div class="bg-white p-4 rounded-xl shadow-sm flex items-start space-x-3">
          <div class="bg-orange-100 p-2 rounded-full text-orange-600">
             <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
          </div>
          <div>
             <div class="font-bold text-gray-900 text-sm">John Doe <span class="text-gray-500 font-normal ml-2">1234567890</span></div>
             <div class="text-sm text-gray-600 mt-1">123 Market St, San Francisco, CA</div>
          </div>
       </div>

       <!-- Product -->
       <div class="bg-white p-4 rounded-xl shadow-sm">
          <div class="flex items-center space-x-2 mb-3 border-b border-gray-50 pb-2">
             <span class="text-xs font-bold text-gray-900">Store Name</span>
             <svg class="w-3 h-3 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path></svg>
          </div>
          <div class="flex space-x-3 mb-4">
             <div class="w-20 h-20 bg-gray-100 rounded-lg flex-shrink-0"></div>
             <div class="flex-1">
                <h3 class="text-sm font-medium text-gray-900 line-clamp-2">Product Name</h3>
                <p class="text-xs text-gray-500 mt-1">Variant: Default</p>
                <div class="flex justify-between mt-2">
                   <span class="text-sm font-bold text-gray-900">$24.99</span>
                   <span class="text-xs text-gray-500">x1</span>
                </div>
             </div>
          </div>
          <div class="flex justify-end border-t border-gray-50 pt-3">
             <span class="text-xs text-gray-500 mr-2">Order Total:</span>
             <span class="text-sm font-bold text-red-600">$24.99</span>
          </div>
       </div>

       <!-- Tracking Link -->
       <button 
         id="order-tracking-link" 
         class="w-full bg-white border border-gray-200 rounded-xl p-4 flex items-center justify-between shadow-sm hover:border-red-500 transition-colors group"
         @click="handleGoTracking"
       >
          <div class="flex items-center space-x-3">
             <div class="bg-blue-100 p-2 rounded-full text-blue-600">
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"></path></svg>
             </div>
             <span class="font-bold text-gray-900 group-hover:text-red-600">Track Order</span>
          </div>
          <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path></svg>
       </button>
    </div>

    <!-- Bottom Bar -->
    <div class="fixed bottom-0 left-0 w-full bg-white border-t border-gray-200 p-4 shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.05)] z-30 flex justify-end space-x-3">
       <button class="px-6 py-2 border border-gray-300 rounded-full text-sm font-medium text-gray-600 hover:bg-gray-50">Cancel Order</button>
       <button class="px-6 py-2 bg-red-600 text-white rounded-full text-sm font-bold hover:bg-red-700 shadow-lg">Pay Now</button>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'ORDER_DETAIL',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const handleBackOrders = async () => {
       signatureStore.currentPageId = 'ORDERS_LIST'
       await router.push({ name: 'ORDERS_LIST' })
    }

    const handleGoTracking = async () => {
       signatureStore.currentPageId = 'ORDER_TRACKING'
       await router.push({ name: 'ORDER_TRACKING' })
    }

    return {
       handleBackOrders,
       handleGoTracking
    }
  }
}
</script>