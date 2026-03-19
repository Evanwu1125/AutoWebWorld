<template>
  <div class="min-h-screen bg-gray-50 pb-24 font-sans">
    <!-- Header -->
    <header class="bg-white shadow-sm px-4 py-3 flex items-center sticky top-0 z-20">
      <button 
        id="checkout-back-cart" 
        class="p-2 -ml-2 hover:bg-gray-100 rounded-full transition-colors"
        @click="handleBackCart"
      >
        <svg class="w-6 h-6 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
      </button>
      <h1 class="text-lg font-bold text-gray-900 ml-2">Checkout</h1>
    </header>

    <!-- Shipping Address -->
    <div class="mt-4 bg-white p-4 shadow-sm mb-2 cursor-pointer" @click="handleSelectAddress('addr_1')">
       <div class="flex justify-between items-start">
          <div class="flex items-center space-x-2 mb-2">
             <div class="bg-red-100 p-1.5 rounded-full text-red-600">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
             </div>
             <h2 class="font-bold text-gray-900">Shipping Address</h2>
          </div>
          <button class="text-red-600 text-sm font-medium">Edit</button>
       </div>
       
       <div id="checkout-address-1" class="pl-8" :class="signatureStore.selected_shipping_address_id ? 'opacity-100' : 'opacity-60'">
          <div v-if="signatureStore.selected_shipping_address_id">
             <div class="font-bold text-gray-800">John Doe <span class="text-gray-500 font-normal text-sm ml-2">+1 234 567 890</span></div>
             <div class="text-sm text-gray-600 mt-1">123 Market Street, Suite 456</div>
             <div class="text-sm text-gray-600">San Francisco, CA 94105, US</div>
          </div>
          <div v-else class="text-gray-500 italic">
             Tap to select shipping address
          </div>
       </div>
    </div>

    <!-- Payment Method -->
    <div class="bg-white p-4 shadow-sm mb-2">
       <h2 class="font-bold text-gray-900 mb-3 flex items-center">
          <svg class="w-5 h-5 text-green-600 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z"></path></svg>
          Payment Method
       </h2>
       
       <div id="checkout-payment-dropdown" class="relative group w-full">
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
               id="payment-option-card" 
               class="px-4 py-3 hover:bg-gray-50 cursor-pointer flex items-center border-b border-gray-100"
               @click="handleSelectPayment('card')"
             >
                <span class="text-2xl mr-3">💳</span>
                <div>
                   <div class="font-medium text-gray-900">Credit/Debit Card</div>
                   <div class="text-xs text-gray-500">Visa, Mastercard, Amex</div>
                </div>
             </div>
             <div 
               id="payment-option-paypal" 
               class="px-4 py-3 hover:bg-gray-50 cursor-pointer flex items-center"
               @click="handleSelectPayment('paypal')"
             >
                <span class="text-2xl mr-3">🅿️</span>
                <div>
                   <div class="font-medium text-gray-900">PayPal</div>
                   <div class="text-xs text-gray-500">Safe, fast payment</div>
                </div>
             </div>
          </div>
       </div>
    </div>

    <!-- Order Summary -->
    <div class="bg-white p-4 shadow-sm mb-20">
       <h2 class="font-bold text-gray-900 mb-3">Order Summary</h2>
       <div class="space-y-2 text-sm">
          <div class="flex justify-between text-gray-600">
             <span>Subtotal</span>
             <span>$129.99</span>
          </div>
          <div class="flex justify-between text-gray-600">
             <span>Shipping</span>
             <span class="text-green-600 font-medium">Free</span>
          </div>
          <div class="flex justify-between text-gray-600">
             <span>Tax</span>
             <span>$10.40</span>
          </div>
          <div class="border-t border-gray-100 pt-2 mt-2 flex justify-between items-center">
             <span class="font-bold text-gray-900 text-base">Total</span>
             <span class="font-black text-red-600 text-xl">$140.39</span>
          </div>
       </div>
    </div>

    <!-- Place Order Bar -->
    <div class="fixed bottom-0 left-0 w-full bg-white border-t border-gray-200 p-4 shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.05)] z-30">
       <button 
         id="checkout-place-order-button"
         class="w-full bg-red-600 text-white font-bold py-3 rounded-full shadow-lg hover:bg-red-700 active:scale-95 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
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
  name: 'CART_CHECKOUT',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const canPlaceOrder = computed(() => {
       return signatureStore.selected_shipping_address_id && signatureStore.selected_payment_method
    })

    const handleBackCart = async () => {
       signatureStore.currentPageId = 'CART_PAGE'
       await router.push({ name: 'CART_PAGE' })
    }

    const handleSelectAddress = (id) => {
       signatureStore.selected_shipping_address_id = id
    }

    const handleSelectPayment = (method) => {
       signatureStore.selected_payment_method = method
    }

    const handlePlaceOrder = async () => {
       signatureStore.order_id = `ORD-${Date.now()}`
       signatureStore.success_message = 'Payment Successful!'
       signatureStore.currentPageId = 'CHECKOUT_FROM_CART_SUCCESS'
       await router.push({ name: 'CHECKOUT_FROM_CART_SUCCESS' })
    }

    return {
       signatureStore,
       canPlaceOrder,
       handleBackCart,
       handleSelectAddress,
       handleSelectPayment,
       handlePlaceOrder
    }
  }
}
</script>