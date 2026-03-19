<template>
  <div class="checkout-page min-h-screen bg-gray-50 flex flex-col">
    <header class="bg-white border-b p-4 sticky top-0 z-30">
      <div class="max-w-3xl mx-auto flex justify-center">
         <div class="font-bold text-xl text-[#0071DC] flex items-center gap-2">
            <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" /></svg>
            Walmart Checkout
         </div>
      </div>
    </header>

    <main class="flex-1 max-w-3xl mx-auto w-full p-4 md:p-8">
      <div class="bg-white rounded-xl shadow-sm overflow-hidden">
        <div class="flex border-b">
           <div class="flex-1 py-3 text-center text-green-600 font-bold flex items-center justify-center gap-1">
             <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" /></svg>
             1. Shipping
           </div>
           <div class="flex-1 py-3 text-center text-green-600 font-bold flex items-center justify-center gap-1">
             <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" /></svg>
             2. Payment
           </div>
           <div class="flex-1 py-3 text-center text-blue-600 font-bold border-b-2 border-blue-600">3. Review</div>
        </div>

        <div class="p-6 md:p-8 space-y-6">
           <h2 class="text-2xl font-bold mb-6">Review your order</h2>

           <!-- Summary Cards -->
           <div class="grid md:grid-cols-2 gap-4">
             <div class="p-4 bg-gray-50 rounded-lg border">
               <h3 class="font-bold text-sm text-gray-500 uppercase tracking-wide mb-2">Shipping To</h3>
               <div class="font-medium">{{ store.shipping_full_name }}</div>
               <div>{{ store.shipping_address_line1 }}</div>
               <div>{{ store.shipping_city }}, {{ store.shipping_zip }}</div>
             </div>
             <div class="p-4 bg-gray-50 rounded-lg border">
               <h3 class="font-bold text-sm text-gray-500 uppercase tracking-wide mb-2">Payment</h3>
               <div class="flex items-center gap-2">
                 <svg class="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z" /></svg>
                 <span class="font-medium">Ending in {{ store.card_number ? store.card_number.slice(-4) : 'xxxx' }}</span>
               </div>
               <div class="text-sm text-gray-500 mt-1">Exp: {{ store.card_expiry }}</div>
             </div>
           </div>
           
           <!-- Terms Checkbox -->
           <div class="py-4">
              <label class="flex items-start gap-3 cursor-pointer p-4 border rounded-lg hover:bg-gray-50 transition-colors">
                <input 
                  id="review-accept-terms-checkbox"
                  type="checkbox" 
                  v-model="termsAccepted"
                  @change="handleTerms"
                  class="mt-1 rounded text-blue-600 w-5 h-5 focus:ring-blue-500"
                />
                <span class="text-sm text-gray-700">
                  I accept the <span class="text-blue-600 underline">Terms of Use</span> and <span class="text-blue-600 underline">Privacy Policy</span>. I agree that my order will be processed according to these terms.
                </span>
              </label>
           </div>

           <!-- Actions -->
           <div class="pt-6 border-t flex items-center justify-between">
              <button 
                id="review-back-to-payment"
                @click="handleBackToPayment"
                class="text-gray-600 font-medium hover:text-[#0071DC] hover:underline"
              >
                &larr; Back to Payment
              </button>
              
              <!-- Both buttons exist in DOM with valid bbox, overlapped via absolute positioning -->
              <div class="relative" style="min-height: 48px;">
                <button 
                  id="place-order-delivery-button"
                  @click="handlePlaceOrderDelivery"
                  :disabled="!termsAccepted"
                  :class="[
                    'bg-[#0071DC] text-white font-bold py-3 px-8 rounded-full shadow-md transition-all',
                    store.shipping_method === 'delivery' ? [
                      'relative z-10',
                      termsAccepted ? 'hover:bg-[#005bb5]' : 'opacity-50 cursor-not-allowed'
                    ] : 'absolute top-0 right-0 opacity-0 pointer-events-none'
                  ]"
                >
                  Place Order (Delivery)
                </button>
                <button 
                  id="place-order-pickup-button"
                  @click="handlePlaceOrderPickup"
                  :disabled="!termsAccepted"
                  :class="[
                    'bg-[#0071DC] text-white font-bold py-3 px-8 rounded-full shadow-md transition-all',
                    store.shipping_method !== 'delivery' ? [
                      'relative z-10',
                      termsAccepted ? 'hover:bg-[#005bb5]' : 'opacity-50 cursor-not-allowed'
                    ] : 'absolute top-0 right-0 opacity-0 pointer-events-none'
                  ]"
                >
                  Place Order (Pickup)
                </button>
              </div>
           </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'CHECKOUT_REVIEW',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const termsAccepted = ref(false)

    const handleTerms = () => {
      // FSM: ACT_REVIEW_ACCEPT_TERMS
      store.review_terms_accepted = termsAccepted.value
    }

    const handlePlaceOrderDelivery = async () => {
      // FSM: ACT_REVIEW_PLACE_ORDER_DELIVERY
      // Simulate order placement
      store.order_id = 'ORD-' + Math.floor(Math.random() * 10000)
      store.cart_items = [] // Clear cart after order

      store.currentPageId = 'CHECKOUT_DELIVERY_SUCCESS'
      await router.push({ name: 'CHECKOUT_DELIVERY_SUCCESS' })
    }

    const handlePlaceOrderPickup = async () => {
      // FSM: ACT_REVIEW_PLACE_ORDER_PICKUP
      store.order_id = 'ORD-' + Math.floor(Math.random() * 10000)
      store.cart_items = [] 

      store.currentPageId = 'CHECKOUT_PICKUP_SUCCESS'
      await router.push({ name: 'CHECKOUT_PICKUP_SUCCESS' })
    }

    const handleBackToPayment = async () => {
      // FSM: ACT_REVIEW_BACK_TO_PAYMENT
      store.currentPageId = 'CHECKOUT_PAYMENT'
      await router.push({ name: 'CHECKOUT_PAYMENT' })
    }

    return {
      store,
      termsAccepted,
      handleTerms,
      handlePlaceOrderDelivery,
      handlePlaceOrderPickup,
      handleBackToPayment
    }
  }
}
</script>