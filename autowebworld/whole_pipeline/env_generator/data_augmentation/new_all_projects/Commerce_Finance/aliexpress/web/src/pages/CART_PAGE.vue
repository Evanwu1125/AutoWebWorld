<template>
  <div class="min-h-screen bg-gray-50 pb-24 font-sans">
    <!-- Header -->
    <header class="bg-white shadow-sm px-4 py-3 flex items-center sticky top-0 z-20">
      <button 
        id="cart-back-home" 
        class="p-2 -ml-2 hover:bg-gray-100 rounded-full transition-colors"
        @click="handleBackHome"
      >
        <svg class="w-6 h-6 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>
      </button>
      <h1 class="text-lg font-bold text-gray-900 ml-2">Shopping Cart ({{ cartItems.length }})</h1>
    </header>

    <!-- Cart Items -->
    <div id="cart-items-container" class="container mx-auto px-4 py-4 space-y-4">
      <div v-if="cartItems.length === 0" class="text-center py-12">
         <div class="text-6xl mb-4">🛒</div>
         <h3 class="text-lg font-bold text-gray-900">Your cart is empty</h3>
         <p class="text-gray-500 mb-6">Time to start shopping!</p>
      </div>

      <div 
        v-for="item in cartItems" 
        :key="item.id"
        :class="[
          'bg-white rounded-xl p-4 shadow-sm flex space-x-4',
          `data-id-${item.id}`,
          'cart-item-row-visible'
        ]"
        @click="handleOpenItem(item)"
      >
        <!-- Checkbox -->
        <div class="flex items-center">
           <input type="checkbox" class="w-5 h-5 text-red-600 rounded border-gray-300 focus:ring-red-500" checked @click.stop />
        </div>
        
        <!-- Image -->
        <div class="w-24 h-24 bg-gray-100 rounded-lg overflow-hidden flex-shrink-0">
           <img :src="item.image" class="w-full h-full object-cover" />
        </div>
        
        <!-- Details -->
        <div class="flex-1 min-w-0 flex flex-col justify-between py-1">
           <div>
              <h3 class="text-sm font-medium text-gray-900 line-clamp-2 mb-1">{{ item.name }}</h3>
              <p class="text-xs text-gray-500 bg-gray-100 inline-block px-1.5 py-0.5 rounded">{{ item.selectedSku }}</p>
           </div>
           <div class="flex justify-between items-end">
              <span class="text-lg font-bold text-red-600">${{ item.price }}</span>
              <div class="flex items-center border border-gray-200 rounded bg-white" @click.stop>
                 <button class="px-2 py-0.5 text-gray-600 hover:bg-gray-50 border-r border-gray-200">-</button>
                 <span class="px-2 text-sm font-medium text-gray-900">{{ item.quantity || 1 }}</span>
                 <button class="px-2 py-0.5 text-gray-600 hover:bg-gray-50 border-l border-gray-200">+</button>
              </div>
           </div>
        </div>
      </div>
    </div>

    <!-- Bottom Checkout Bar -->
    <div v-if="cartItems.length > 0" class="fixed bottom-0 left-0 w-full bg-white border-t border-gray-200 p-4 shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.05)] z-30">
       <div class="container mx-auto flex justify-between items-center">
          <div class="flex items-center space-x-2">
             <input type="checkbox" class="w-5 h-5 text-red-600 rounded border-gray-300 focus:ring-red-500" checked />
             <span class="text-sm text-gray-600 font-medium">All</span>
          </div>
          <div class="flex items-center space-x-4">
             <div class="text-right">
                <div class="text-xs text-gray-500">Total:</div>
                <div class="text-lg font-black text-red-600">${{ total.toFixed(2) }}</div>
             </div>
             <button 
               id="cart-checkout-button"
               class="bg-red-600 text-white font-bold py-2.5 px-8 rounded-full shadow-lg hover:bg-red-700 active:scale-95 transition-all"
               @click="handleCheckout"
             >
               Checkout ({{ cartItems.length }})
             </button>
          </div>
       </div>
    </div>

  </div>
</template>

<script>
import { computed, watch, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'CART_PAGE',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()
    
    // Get cart items from dataStore
    const cartItems = computed(() => {
       return dataStore.cart_items.map(cartItem => {
         const product = dataStore.products.find(p => p.id === cartItem.productId)
         if (!product) return null
         return {
           ...product,
           quantity: cartItem.quantity,
           selectedSku: cartItem.selectedSku
         }
       }).filter(item => item !== null)
    })

    const total = computed(() => {
       return cartItems.value.reduce((sum, item) => sum + (item.price * item.quantity), 0)
    })

    const handleBackHome = async () => {
       signatureStore.currentPageId = 'HOME'
       await router.push({ name: 'HOME' })
    }

    const handleOpenItem = async (item) => {
       signatureStore.CART_PAGE_viewport_anchor_id = null // Clear flag
       signatureStore.selected_item_id = item.id
       signatureStore.currentPageId = 'PRODUCT_DETAIL'
       await router.push({ name: 'PRODUCT_DETAIL' })
    }

    const handleCheckout = async () => {
       // Ensure cart items in signature
       signatureStore.cart_items = cartItems.value
       signatureStore.currentPageId = 'CART_CHECKOUT'
       await router.push({ name: 'CART_CHECKOUT' })
    }

    // Scroll handler
    watch(() => signatureStore.CART_PAGE_viewport_anchor_id, async (newId) => {
      if (newId) {
        await nextTick()
        const element = document.querySelector(`.data-id-${newId}`)
        if (element) {
          element.scrollIntoView({ behavior: 'smooth' })
        }
      }
    })

    return {
       cartItems,
       total,
       handleBackHome,
       handleOpenItem,
       handleCheckout
    }
  }
}
</script>