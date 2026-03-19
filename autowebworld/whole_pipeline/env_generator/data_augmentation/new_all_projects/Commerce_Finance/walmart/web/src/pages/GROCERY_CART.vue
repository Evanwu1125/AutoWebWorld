<template>
  <div class="grocery-cart-page min-h-screen bg-gray-50 flex flex-col">
    <header class="bg-[#2A8703] text-white p-4 sticky top-0 z-30 shadow-md">
      <div class="max-w-7xl mx-auto flex items-center justify-between">
         <div id="grocery-cart-logo-home" @click="handleGoHome" class="font-bold text-xl cursor-pointer flex items-center gap-2">
            <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" /></svg>
            Walmart Grocery
         </div>
         <h1 class="text-lg font-medium">Your Basket</h1>
      </div>
    </header>

    <main class="flex-1 max-w-7xl mx-auto w-full p-4 md:p-8">
      <div v-if="cartItems.length > 0" class="flex flex-col lg:flex-row gap-8">
        
        <!-- Cart Items -->
        <div class="flex-1 space-y-4">
           <div class="bg-white rounded-xl shadow-sm p-4 border border-gray-100">
             <div class="flex justify-between items-center mb-4 border-b pb-2">
               <h2 class="font-bold text-xl text-gray-800">Basket ({{ cartItems.length }} items)</h2>
             </div>
             
             <div class="space-y-4">
               <div v-for="(item, index) in cartDetails" :key="index" class="flex gap-4 py-4 border-b last:border-0 items-center">
                  <div class="w-16 h-16 bg-gray-50 rounded-lg p-1 border border-gray-100 flex-shrink-0">
                    <img :src="item.image" :alt="item.name" class="w-full h-full object-cover rounded mix-blend-multiply" />
                  </div>
                  
                  <div class="flex-1">
                    <div 
                      id="grocery-cart-back-to-product"
                      @click="handleBackToProduct(item.id)"
                      class="font-medium text-gray-900 cursor-pointer hover:text-[#2A8703]"
                    >
                      {{ item.name }}
                    </div>
                    <div class="text-sm text-gray-500">${{ item.price.toFixed(2) }} / {{ item.unit }}</div>
                  </div>
                  
                  <div class="font-bold text-lg">${{ (item.price * item.qty).toFixed(2) }}</div>
               </div>
             </div>
           </div>
        </div>

        <!-- Summary -->
        <div class="w-full lg:w-80 flex-shrink-0">
          <div class="bg-white rounded-xl shadow-sm p-6 sticky top-24 border border-gray-100">
             <div class="flex justify-between text-lg font-bold mb-6">
               <span>Subtotal</span>
               <span>${{ total.toFixed(2) }}</span>
             </div>
             
             <button 
               id="grocery-cart-schedule-button"
               @click="handleProceed"
               class="w-full bg-[#2A8703] text-white font-bold py-3 rounded-full hover:bg-[#237002] shadow-md transition-all transform hover:-translate-y-0.5"
             >
               Schedule Delivery
             </button>
             
             <div class="mt-4 text-xs text-center text-gray-500">
               Proceed to choose your delivery time slot.
             </div>
          </div>
        </div>

      </div>

      <!-- Empty Cart -->
      <div v-else class="text-center py-20 bg-white rounded-xl shadow-sm">
         <div class="text-6xl mb-4">🧺</div>
         <h2 class="text-2xl font-bold mb-2">Basket is empty</h2>
         <p class="text-gray-500 mb-8">Add fresh groceries to get started.</p>
         <button 
           @click="handleGoHome"
           class="bg-[#2A8703] text-white font-bold py-3 px-8 rounded-full hover:bg-[#237002] transition-colors"
         >
           Start Shopping
         </button>
      </div>
    </main>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'GROCERY_CART',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const cartItems = computed(() => store.grocery_cart_items)
    
    const cartDetails = computed(() => {
      return cartItems.value.map(item => {
        const product = dataStore.groceries.find(p => p.id === item.id) || { name: 'Unknown', price: 0, image: '', unit: '' }
        return { ...product, qty: item.qty || 1 }
      })
    })

    const total = computed(() => cartDetails.value.reduce((acc, item) => acc + item.price * item.qty, 0))

    const handleProceed = async () => {
      // FSM: ACT_GROCERY_CART_PROCEED_TO_SCHEDULING
      store.currentPageId = 'GROCERY_DELIVERY_SCHEDULING'
      await router.push({ name: 'GROCERY_DELIVERY_SCHEDULING' })
    }

    const handleGoHome = async () => {
      // FSM: ACT_GROCERY_CART_BACK_TO_HOME
      store.currentPageId = 'HOME'
      await router.push({ name: 'HOME' })
    }

    const handleBackToProduct = async (itemId) => {
      // FSM: ACT_GROCERY_CART_BACK_TO_PRODUCT
      store.selected_product_id = itemId
      store.currentPageId = 'GROCERY_PRODUCT_DETAIL'
      await router.push({ name: 'GROCERY_PRODUCT_DETAIL', params: { id: itemId } })
    }

    return {
      cartItems,
      cartDetails,
      total,
      handleProceed,
      handleGoHome,
      handleBackToProduct
    }
  }
}
</script>