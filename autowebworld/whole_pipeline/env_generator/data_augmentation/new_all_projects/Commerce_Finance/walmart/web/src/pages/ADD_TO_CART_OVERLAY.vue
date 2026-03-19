<template>
  <div class="add-to-cart-overlay-page fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
    <div class="bg-white w-full max-w-2xl rounded-2xl shadow-2xl overflow-hidden animate-fade-in-up">
      <!-- Header -->
      <div class="p-4 border-b flex justify-between items-center bg-green-50">
        <div class="flex items-center gap-2 text-green-700 font-bold text-lg">
          <div class="bg-green-100 p-1 rounded-full">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7" /></svg>
          </div>
          Added to cart
        </div>
        <button 
          id="overlay-close" 
          @click="handleBackToProduct"
          class="text-gray-400 hover:text-gray-600 p-1 rounded-full hover:bg-white transition-colors"
        >
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" /></svg>
        </button>
      </div>

      <!-- Content -->
      <div class="p-6 md:p-8">
        <div class="flex flex-col md:flex-row gap-6">
          <!-- Product Preview -->
          <div class="flex items-start gap-4 flex-1">
             <div class="w-24 h-24 bg-gray-50 rounded-lg p-2 border border-gray-100 flex-shrink-0">
               <img :src="latestItem.image" :alt="latestItem.name" class="w-full h-full object-contain mix-blend-multiply" />
             </div>
             <div>
               <h3 class="font-medium text-gray-900 mb-1 line-clamp-2">{{ latestItem.name }}</h3>
               <div class="font-bold text-lg text-gray-900">${{ latestItem.price.toFixed(2) }}</div>
             </div>
          </div>

          <!-- Actions -->
          <div class="flex flex-col gap-3 w-full md:w-64">
            <button 
              id="view-cart-button"
              @click="handleViewCart"
              class="w-full bg-[#0071DC] text-white font-bold py-3 rounded-full hover:bg-[#005bb5] shadow-md transition-transform hover:-translate-y-0.5"
            >
              View Cart ({{ cartCount }} items)
            </button>
            <button 
              id="continue-shopping-button"
              @click="handleContinueShopping"
              class="w-full bg-white text-gray-700 font-bold py-3 rounded-full border border-gray-300 hover:bg-gray-50 transition-colors"
            >
              Continue Shopping
            </button>
          </div>
        </div>

        <!-- Recommendations (Decorative) -->
        <div class="mt-8 pt-6 border-t">
          <h4 class="font-bold text-gray-800 mb-4">Customers also bought</h4>
          <div class="grid grid-cols-3 gap-4">
             <div v-for="rec in recommendations" :key="rec.id" class="p-2 border rounded-lg hover:shadow-md cursor-pointer transition-shadow">
               <div class="aspect-square bg-gray-50 mb-2 rounded flex items-center justify-center">
                 <img :src="rec.image" class="max-h-full max-w-full object-contain mix-blend-multiply" />
               </div>
               <div class="text-sm font-medium truncate">{{ rec.name }}</div>
               <div class="font-bold text-sm">${{ rec.price.toFixed(2) }}</div>
             </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'ADD_TO_CART_OVERLAY',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    // Get latest added item
    const cartItems = computed(() => store.cart_items)
    const cartCount = computed(() => cartItems.value.length)
    
    const latestItemData = computed(() => {
      if (cartItems.value.length === 0) return null
      const last = cartItems.value[cartItems.value.length - 1]
      return dataStore.electronics.find(p => p.id === last.id)
    })

    const latestItem = computed(() => latestItemData.value || { name: 'Item', price: 0, image: '' })

    const recommendations = computed(() => {
      // Mock recommendations (first 3 items not in cart)
      return dataStore.electronics.filter(p => p.id !== latestItem.value.id).slice(0, 3)
    })

    // Handlers
    const handleViewCart = async () => {
      // FSM: ACT_OVERLAY_VIEW_CART
      store.currentPageId = 'CART'
      await router.push({ name: 'CART' })
    }

    const handleContinueShopping = async () => {
      // FSM: ACT_OVERLAY_CONTINUE_SHOPPING
      store.currentPageId = 'ELECTRONICS_CATEGORY'
      await router.push({ name: 'ELECTRONICS_CATEGORY' })
    }

    const handleBackToProduct = async () => {
      // FSM: ACT_OVERLAY_BACK_TO_PRODUCT
      store.currentPageId = 'PRODUCT_DETAIL'
      await router.push({ name: 'PRODUCT_DETAIL', params: { id: latestItem.value.id } })
    }

    return {
      cartCount,
      latestItem,
      recommendations,
      handleViewCart,
      handleContinueShopping,
      handleBackToProduct
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
  animation: fadeInUp 0.3s ease-out forwards;
}
</style>