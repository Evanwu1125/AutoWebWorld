<template>
  <div class="product-detail-page min-h-screen bg-white flex flex-col">
    <header class="bg-white border-b sticky top-0 z-30">
      <div class="max-w-7xl mx-auto px-4 py-4 flex items-center gap-4">
        <div 
          id="breadcrumb-electronics" 
          @click="handleBackToElectronics"
          class="cursor-pointer text-gray-500 hover:text-[#0071DC] flex items-center gap-1 text-sm"
        >
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" /></svg>
          Back to Electronics
        </div>
      </div>
    </header>

    <main v-if="product" class="flex-1 max-w-7xl mx-auto w-full p-4 md:p-8">
      <div class="grid md:grid-cols-2 gap-12">
        
        <!-- Product Image Gallery -->
        <div class="space-y-4">
          <div class="aspect-square bg-gray-50 rounded-2xl overflow-hidden flex items-center justify-center p-8">
             <img :src="product.image" :alt="product.name" class="max-w-full max-h-full object-contain mix-blend-multiply hover:scale-105 transition-transform duration-500" />
          </div>
          <div class="grid grid-cols-4 gap-4">
             <div class="aspect-square bg-gray-50 rounded-lg border-2 border-blue-500 cursor-pointer p-2">
                <img :src="product.image" class="w-full h-full object-contain mix-blend-multiply" />
             </div>
             <!-- Mock thumbnails -->
             <div v-for="i in 3" :key="i" class="aspect-square bg-gray-50 rounded-lg border border-transparent hover:border-gray-300 cursor-pointer p-2 opacity-60">
                <img :src="product.image" class="w-full h-full object-contain mix-blend-multiply grayscale" />
             </div>
          </div>
        </div>

        <!-- Product Info & Actions -->
        <div>
          <div class="mb-6 border-b pb-6">
             <h1 class="text-3xl font-bold text-gray-900 mb-2">{{ product.name }}</h1>
             <div class="flex items-center gap-4 mb-4">
               <div class="flex text-sm items-center gap-1">
                 <span class="font-bold">{{ product.rating }}</span>
                 <div class="flex text-yellow-400">★★★★☆</div>
                 <span class="text-gray-500 underline cursor-pointer">1,245 reviews</span>
               </div>
               <div class="text-xs font-medium px-2 py-0.5 bg-gray-100 rounded text-gray-600">Best Seller</div>
             </div>
             <div class="text-4xl font-bold text-[#2A8703]">${{ product.price.toFixed(2) }}</div>
          </div>

          <!-- Quantity -->
          <div class="mb-6">
            <label class="block text-sm font-semibold text-gray-700 mb-2">Quantity</label>
            <div class="flex items-center">
              <input 
                id="quantity-input"
                type="number" 
                v-model.number="quantity"
                @input="handleSetQuantity"
                min="1"
                class="w-24 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none font-medium text-center"
              />
            </div>
          </div>

          <!-- Shipping Options -->
          <div class="mb-6 space-y-3">
             <div class="relative">
                <button 
                  id="shipping-options-dropdown"
                  @click="showShipping = !showShipping"
                  class="w-full flex items-center justify-between px-4 py-3 border border-gray-300 rounded-lg hover:border-gray-400 bg-white text-left"
                >
                  <span class="flex items-center gap-2">
                    <svg class="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" /></svg>
                    <span v-if="shippingMethod === 'delivery'">Delivery to <strong>94043</strong></span>
                    <span v-else>Pickup</span>
                  </span>
                  <svg class="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" /></svg>
                </button>
                
                <div v-if="showShipping" class="absolute top-full left-0 right-0 mt-2 bg-white rounded-lg shadow-xl border border-gray-100 z-20">
                   <div 
                     id="shipping-option-delivery" 
                     @click="handleSelectShipping('delivery')"
                     class="px-4 py-3 hover:bg-gray-50 cursor-pointer flex items-center justify-between border-b border-gray-100"
                   >
                     <div>
                       <div class="font-bold text-sm">Delivery</div>
                       <div class="text-xs text-green-600">Tomorrow, Free</div>
                     </div>
                     <span v-if="shippingMethod === 'delivery'" class="text-blue-600">✓</span>
                   </div>
                   <div 
                     id="shipping-option-pickup" 
                     @click="handleSelectShipping('pickup')"
                     class="px-4 py-3 hover:bg-gray-50 cursor-pointer flex items-center justify-between"
                   >
                     <div>
                       <div class="font-bold text-sm">Pickup</div>
                       <div class="text-xs text-gray-500">Select a store</div>
                     </div>
                     <span v-if="shippingMethod === 'pickup'" class="text-blue-600">✓</span>
                   </div>
                </div>
             </div>

             <!-- Pickup Store Menu (Hover) -->
             <div 
               id="pickup-store-menu"
               class="relative group"
               v-if="shippingMethod === 'pickup'"
             >
                <div class="w-full px-4 py-3 bg-gray-50 border border-dashed border-gray-300 rounded-lg text-sm text-gray-600 cursor-pointer flex justify-between items-center hover:bg-gray-100">
                   <span>{{ selectedStoreName || 'Select a store for pickup' }}</span>
                   <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" /><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" /></svg>
                </div>
                
                <div class="absolute top-full left-0 right-0 mt-1 bg-white rounded-lg shadow-lg border border-gray-100 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none group-hover:pointer-events-auto z-20">
                   <div class="p-2 text-xs font-bold text-gray-500 uppercase tracking-wider">Nearby Stores</div>
                   <div 
                     v-for="(store, idx) in stores" 
                     :key="store.id"
                     :class="`store-option-${idx+1}`"
                     @click="handleSelectStore(store.id)"
                     class="px-4 py-3 hover:bg-blue-50 cursor-pointer border-t border-gray-100"
                   >
                     <div class="font-bold text-sm text-gray-800">{{ store.name }}</div>
                     <div class="text-xs text-gray-500">{{ store.address }}</div>
                   </div>
                </div>
             </div>
          </div>

          <!-- CTA Buttons -->
          <div class="flex gap-4 mt-8">
            <button 
              id="add-to-cart-button"
              @click="handleAddToCart"
              class="flex-1 bg-[#0071DC] text-white font-bold py-3 px-6 rounded-full shadow-lg hover:bg-[#005bb5] transition-all transform hover:-translate-y-0.5"
            >
              Add to Cart
            </button>
            <button 
              id="buy-now-button"
              @click="handleBuyNow"
              class="flex-1 bg-[#FFC220] text-gray-900 font-bold py-3 px-6 rounded-full shadow-lg hover:bg-[#ffcf4d] transition-all transform hover:-translate-y-0.5"
            >
              Buy Now
            </button>
          </div>
          
          <div class="mt-8 text-sm text-gray-500 space-y-2">
            <div class="flex items-center gap-2">
              <svg class="w-5 h-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
              Free 90-day returns
            </div>
            <div class="flex items-center gap-2">
              <svg class="w-5 h-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" /></svg>
              Secure transaction
            </div>
          </div>
        </div>
      </div>
      
      <!-- Description Section -->
      <div class="mt-16 pt-8 border-t">
        <h2 class="text-2xl font-bold mb-4">About this item</h2>
        <p class="text-gray-700 leading-relaxed max-w-4xl">{{ product.description }}</p>
      </div>
    </main>

    <!-- Loading State -->
    <div v-else class="flex-1 flex items-center justify-center">
       <div class="animate-spin rounded-full h-12 w-12 border-4 border-blue-200 border-t-blue-600"></div>
    </div>

  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'PRODUCT_DETAIL',
  setup() {
    const route = useRoute()
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const productId = route.params.id || store.selected_product_id
    const product = computed(() => dataStore.electronics.find(p => p.id === productId))
    
    // States
    const quantity = ref(1)
    const showShipping = ref(false)
    const shippingMethod = ref('delivery')
    const selectedStoreId = ref(null)

    // Derived
    const stores = computed(() => dataStore.stores)
    const selectedStoreName = computed(() => stores.value.find(s => s.id === selectedStoreId.value)?.name)

    // Handlers
    const handleSetQuantity = () => {
      // FSM: ACT_PRODUCT_SET_QUANTITY
      store.selected_quantity = quantity.value
    }

    const handleSelectShipping = (method) => {
      // FSM: ACT_PRODUCT_SELECT_SHIPPING_OPTION
      shippingMethod.value = method
      showShipping.value = false
      store.selected_shipping_option = method
    }

    const handleSelectStore = (storeId) => {
      // FSM: ACT_PRODUCT_SELECT_PICKUP_STORE
      selectedStoreId.value = storeId
      store.selected_pickup_store = storeId
    }

    const handleAddToCart = async () => {
      // FSM: ACT_PRODUCT_ADD_TO_CART
      // Effect: Append to cart_items
      store.cart_items.push({ id: product.value.id, qty: quantity.value })
      
      store.currentPageId = 'ADD_TO_CART_OVERLAY'
      await router.push({ name: 'ADD_TO_CART_OVERLAY' })
    }

    const handleBuyNow = async () => {
      // FSM: ACT_PRODUCT_BUY_NOW
      // Note: Typically Buy Now also adds to cart or creates temp checkout session.
      // FSM effects are empty for this action, implies it just navigates using current product context?
      // Or maybe we should set it as 'selected' for checkout. 
      // But FSM checkouts use 'cart_items'.
      // Wait, CHECKOUT_SHIPPING doesn't check 'cart_items' in preconditions?
      // Let's check FSM: CHECKOUT_SHIPPING actions don't check items.
      // But usually checkout needs items. 
      // We'll push to cart_items implicitly or handle it as a single-item checkout flow if needed.
      // For simplicity, we'll push to cart_items to ensure data exists for checkout review.
      store.cart_items.push({ id: product.value.id, qty: quantity.value })
      
      store.currentPageId = 'CHECKOUT_SHIPPING'
      await router.push({ name: 'CHECKOUT_SHIPPING' })
    }

    const handleBackToElectronics = async () => {
      store.currentPageId = 'ELECTRONICS_CATEGORY'
      await router.push({ name: 'ELECTRONICS_CATEGORY' })
    }

    // Init store with default quantity on mount
    onMounted(() => {
      store.selected_quantity = 1
    })

    return {
      product,
      quantity,
      showShipping,
      shippingMethod,
      stores,
      selectedStoreName,
      handleSetQuantity,
      handleSelectShipping,
      handleSelectStore,
      handleAddToCart,
      handleBuyNow,
      handleBackToElectronics
    }
  }
}
</script>