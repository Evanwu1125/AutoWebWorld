<template>
  <div class="min-h-screen bg-gray-50 font-sans">
    <header class="bg-white shadow-sm px-4 py-3 flex items-center sticky top-0 z-20">
      <button 
        id="tracking-back-order" 
        class="p-2 -ml-2 hover:bg-gray-100 rounded-full transition-colors"
        @click="handleBackOrder"
      >
        <svg class="w-6 h-6 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
      </button>
      <h1 class="text-lg font-bold text-gray-900 ml-2">Tracking</h1>
    </header>

    <div class="p-4">
       <div class="bg-white p-6 rounded-xl shadow-sm">
          <div class="flex justify-between items-start mb-6">
             <div>
                <div class="text-sm text-gray-500">Tracking Number</div>
                <div class="font-mono font-bold text-gray-900 text-lg">TRK-123456789</div>
             </div>
             <button class="text-red-600 text-xs font-bold bg-red-50 px-2 py-1 rounded">COPY</button>
          </div>
          
          <!-- Timeline -->
          <div class="relative border-l-2 border-gray-200 ml-3 space-y-8">
             <div class="relative pl-6">
                <div class="absolute -left-[9px] top-0 w-4 h-4 rounded-full bg-red-600 border-4 border-white shadow-sm"></div>
                <div class="font-bold text-gray-900 text-sm">Order Placed</div>
                <div class="text-xs text-gray-500 mt-1">Oct 24, 10:30 AM</div>
                <div class="text-sm text-gray-600 mt-1">Your order has been received.</div>
             </div>
             <div class="relative pl-6 opacity-50">
                <div class="absolute -left-[9px] top-0 w-4 h-4 rounded-full bg-gray-300 border-4 border-white"></div>
                <div class="font-bold text-gray-900 text-sm">Shipped</div>
                <div class="text-sm text-gray-600 mt-1">Package has left the facility.</div>
             </div>
             <div class="relative pl-6 opacity-50">
                <div class="absolute -left-[9px] top-0 w-4 h-4 rounded-full bg-gray-300 border-4 border-white"></div>
                <div class="font-bold text-gray-900 text-sm">Out for Delivery</div>
                <div class="text-sm text-gray-600 mt-1">Driver is on the way.</div>
             </div>
             <div class="relative pl-6 opacity-50">
                <div class="absolute -left-[9px] top-0 w-4 h-4 rounded-full bg-gray-300 border-4 border-white"></div>
                <div class="font-bold text-gray-900 text-sm">Delivered</div>
                <div class="text-sm text-gray-600 mt-1">Package delivered.</div>
             </div>
          </div>
       </div>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'ORDER_TRACKING',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const handleBackOrder = async () => {
       signatureStore.currentPageId = 'ORDER_DETAIL'
       await router.push({ name: 'ORDER_DETAIL' })
    }

    return {
       handleBackOrder
    }
  }
}
</script>