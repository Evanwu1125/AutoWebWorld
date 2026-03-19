<template>
  <div class="min-h-screen bg-gray-50 pb-24 font-sans">
    <!-- Header -->
    <header class="bg-white shadow-sm px-4 py-3 flex items-center sticky top-0 z-20">
      <button 
        id="buy-now-back-product" 
        class="p-2 -ml-2 hover:bg-gray-100 rounded-full transition-colors"
        @click="handleBackProduct"
      >
        <svg class="w-6 h-6 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
      </button>
      <h1 class="text-lg font-bold text-gray-900 ml-2">Order Confirmation</h1>
    </header>

    <!-- Shipping Address -->
    <div class="mt-4 bg-white p-4 shadow-sm mb-2 cursor-pointer" @click="handleSelectAddress('addr_1')">
       <div class="flex justify-between items-start">
          <div class="flex items-center space-x-2 mb-2">
             <div class="bg-orange-100 p-1.5 rounded-full text-orange-600">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
             </div>
             <h2 class="font-bold text-gray-900">Shipping Address</h2>
          </div>
          <button class="text-red-600 text-sm font-medium">Change</button>
       </div>
       
       <div id="buy-now-address-1" class="pl-8" :class="signatureStore.selected_shipping_address_id ? 'opacity-100' : 'opacity-60'">
          <div v-if="signatureStore.selected_shipping_address_id">
             <div class="font-bold text-gray-800">Jane Smith <span class="text-gray-500 font-normal text-sm ml-2">+1 987 654 321</span></div>
             <div class="text-sm text-gray-600 mt-1">789 Broadway Ave, Apt 101</div>
             <div class="text-sm text-gray-600">New York, NY 10003, US</div>
          </div>
          <div v-else class="text-gray-500 italic">
             Tap to select delivery address
          </div>
       </div>
    </div>

    <!-- Payment Method -->
    <div class="bg-white p-4 shadow-sm mb-2">
       <h2 class="font-bold text-gray-900 mb-3 flex items-center">
          <svg class="w-5 h-5 text-green-600 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z"></path></svg>
          Payment
       </h2>
       
       <div id="buy-now-payment-dropdown" class="relative group w-full">
          <button class="w-full border border-gray-300 rounded-lg px-4 py-3 flex items-center justify-between bg-gray-50 hover:bg-white hover:border-red-500 transition-all">
             <span class="flex items-center" v-if="signatureStore.selected_payment_method === 'card'">
                <span class="text-2xl mr-3">💳</span> Credit/Debit Card
             </span>
             <span class="flex items-center" v-else-if="signatureStore.selected_payment_method === 'paypal'">
                <span class="text-2xl mr-3">🅿️</span> PayPal
             </span>
             <span class="text-gray-500" v-else>Select Payment Method</span>
             
             <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
          </button>
          
          <!-- Dropdown Options -->
          <div class="absolute left-0 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg hidden group-hover:block z-10">
             <div 
               id="buy-now-payment-card" 
               class="px-4 py-3 hover:bg-gray-50 cursor-pointer flex items-center border-b border-gray-100"
               @click="handleSelectPayment('card')"
             >
                <span class="text-2xl mr-3">💳</span>
                <div>
                   <div class="font-medium text-gray-900">Credit/Debit Card</div>
                   <div class="text-xs text-gray-500">Visa, Mastercard</div>
                </div>
             </div>
             <div 
               id="buy-now-payment-paypal" 
               class="px-4 py-3 hover:bg-gray-50 cursor-pointer flex items-center"
               @click="handleSelectPayment('paypal')"
             >
                <span class="text-2xl mr-3">🅿️</span>
                <div>
                   <div class="font-medium text-gray-900">PayPal</div>
                   <div class="text-xs text-gray-500">Quick checkout</div>
                </div>
             </div>
          </div>
       </div>
    </div>

    <!-- Item Summary -->
    <div class="bg-white p-4 shadow-sm mb-20">
       <div class="flex space-x-3 mb-3">
          <div class="w-20 h-20 bg-gray-100 rounded-lg flex-shrink-0"></div>
          <div class="flex-1">
             <h3 class="text-sm font-medium text-gray-900 line-clamp-2">Selected Product Name Here...</h3>
             <div class="mt-1 text-xs text-gray-500">Qty: {{ signatureStore.quantity }}</div>
             <div class="mt-1 font-bold text-red-600">$45.00</div>
          </div>
       </div>
    </div>

    <!-- Place Order Bar -->
    <div class="fixed bottom-0 left-0 w-full bg-white border-t border-gray-200 p-4 shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.05)] z-30 flex items-center justify-between">
       <div class="flex flex-col">
          <span class="text-xs text-gray-500">Total:</span>
          <span class="text-xl font-black text-red-600">$45.00</span>
       </div>
       <button 
         id="buy-now-place-order-button"
         class="w-2/3 bg-red-600 text-white font-bold py-3 rounded-full shadow-lg hover:bg-red-700 active:scale-95 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
         :disabled="!canPlaceOrder"
         @click="handlePlaceOrder"
       >
         Place Order
       </button>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'BUY_NOW_CHECKOUT',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const canPlaceOrder = computed(() => {
       return signatureStore.selected_shipping_address_id && signatureStore.selected_payment_method
    })

    const handleBackProduct = async () => {
       signatureStore.currentPageId = 'PRODUCT_DETAIL'
       await router.push({ name: 'PRODUCT_DETAIL' })
    }

    const handleSelectAddress = (id) => {
       signatureStore.selected_shipping_address_id = id
    }

    const handleSelectPayment = (method) => {
       signatureStore.selected_payment_method = method
    }

    const handlePlaceOrder = async () => {
       signatureStore.order_id = `ORD-${Date.now()}`
       signatureStore.success_message = 'Order Placed Successfully!'
       signatureStore.currentPageId = 'CHECKOUT_BUYNOW_SUCCESS'
       await router.push({ name: 'CHECKOUT_BUYNOW_SUCCESS' })
    }

    return {
       signatureStore,
       canPlaceOrder,
       handleBackProduct,
       handleSelectAddress,
       handleSelectPayment,
       handlePlaceOrder
    }
  }
}
</script>