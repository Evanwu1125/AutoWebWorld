<template>
  <div class="grocery-review-page min-h-screen bg-gray-50 flex flex-col">
    <header class="bg-[#2A8703] text-white p-4 sticky top-0 z-30 shadow-md">
      <div class="max-w-7xl mx-auto flex items-center justify-between">
         <div class="font-bold text-xl flex items-center gap-2">
            <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" /></svg>
            Walmart Grocery
         </div>
         <h1 class="text-lg font-medium">Review Order</h1>
      </div>
    </header>

    <main class="flex-1 max-w-3xl mx-auto w-full p-4 md:p-8">
      <div class="bg-white rounded-xl shadow-sm overflow-hidden">
        <div class="flex border-b">
           <div class="flex-1 py-3 text-center text-[#2A8703] font-bold flex items-center justify-center gap-1">
             <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" /></svg>
             1. Schedule
           </div>
           <div class="flex-1 py-3 text-center text-[#2A8703] font-bold border-b-2 border-[#2A8703]">2. Review</div>
        </div>

        <div class="p-6 md:p-8 space-y-6">
           <h2 class="text-2xl font-bold mb-6 text-gray-900">Review Delivery Details</h2>
           
           <div class="bg-gray-50 p-6 rounded-xl border border-gray-100">
              <div class="mb-4">
                <h3 class="text-sm font-bold text-gray-500 uppercase tracking-wide mb-1">Delivery Time</h3>
                <div class="font-bold text-lg text-gray-900">{{ store.grocery_delivery_slot }}</div>
              </div>
              <div>
                <h3 class="text-sm font-bold text-gray-500 uppercase tracking-wide mb-1">Items</h3>
                <div class="font-medium">{{ store.grocery_cart_items ? store.grocery_cart_items.length : 0 }} items in basket</div>
              </div>
           </div>

           <!-- Terms Checkbox -->
           <div class="py-4">
              <label class="flex items-start gap-3 cursor-pointer p-4 border rounded-lg hover:bg-gray-50 transition-colors">
                <input 
                  id="grocery-review-accept-terms-checkbox"
                  type="checkbox" 
                  v-model="termsAccepted"
                  @change="handleTerms"
                  class="mt-1 rounded text-[#2A8703] w-5 h-5 focus:ring-[#2A8703]"
                />
                <span class="text-sm text-gray-700">
                  I accept the <span class="text-[#2A8703] underline">Terms of Service</span>. I agree to pay the total amount upon delivery confirmation.
                </span>
              </label>
           </div>

           <!-- Actions -->
           <div class="pt-6 border-t flex items-center justify-between">
              <button 
                id="grocery-review-back-to-scheduling"
                @click="handleBackToScheduling"
                class="text-gray-600 font-medium hover:text-[#2A8703] hover:underline"
              >
                &larr; Back to Scheduling
              </button>
              <button 
                id="grocery-place-order-button"
                @click="handlePlaceOrder"
                :disabled="!termsAccepted"
                class="bg-[#2A8703] text-white font-bold py-3 px-8 rounded-full shadow-md hover:bg-[#237002] disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                Place Order
              </button>
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
  name: 'GROCERY_CHECKOUT_REVIEW',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const termsAccepted = ref(false)

    const handleTerms = () => {
      // FSM: ACT_GROCERY_REVIEW_ACCEPT_TERMS
      store.grocery_review_terms_accepted = termsAccepted.value
    }

    const handlePlaceOrder = async () => {
      // FSM: ACT_GROCERY_REVIEW_PLACE_ORDER
      store.order_id = 'GROC-' + Math.floor(Math.random() * 10000)
      store.grocery_cart_items = [] // Clear

      store.currentPageId = 'CHECKOUT_GROCERY_SUCCESS'
      await router.push({ name: 'CHECKOUT_GROCERY_SUCCESS' })
    }

    const handleBackToScheduling = async () => {
      // FSM: ACT_GROCERY_REVIEW_BACK_TO_SCHEDULING
      store.currentPageId = 'GROCERY_DELIVERY_SCHEDULING'
      await router.push({ name: 'GROCERY_DELIVERY_SCHEDULING' })
    }

    return {
      store,
      termsAccepted,
      handleTerms,
      handlePlaceOrder,
      handleBackToScheduling
    }
  }
}
</script>