<template>
  <div class="min-h-screen bg-gray-50 pb-24 relative">
    <!-- Header (Transparent/Overlay) -->
    <div class="absolute top-0 left-0 w-full z-10 p-4 flex justify-between items-center bg-gradient-to-b from-black/30 to-transparent">
      <button 
        id="product-detail-back-list" 
        class="p-2 bg-black/40 text-white rounded-full hover:bg-black/60 backdrop-blur-md transition-colors"
        @click="handleBackList"
      >
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
      </button>
      <div class="flex space-x-3">
         <button class="p-2 bg-black/40 text-white rounded-full hover:bg-black/60 backdrop-blur-md">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"></path></svg>
         </button>
         <button class="p-2 bg-black/40 text-white rounded-full hover:bg-black/60 backdrop-blur-md">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z"></path></svg>
         </button>
      </div>
    </div>

    <!-- Product Images (Gallery) -->
    <div class="aspect-square w-full bg-white relative">
       <img :src="product?.image || '/images/Product.jpg'" class="w-full h-full object-cover" alt="Product Detail" />
       <div class="absolute bottom-4 right-4 bg-black/60 text-white text-xs px-2 py-1 rounded-full font-medium">
          1/5 Photos
       </div>
    </div>

    <!-- Product Info -->
    <div class="bg-white p-4 mb-2 shadow-sm rounded-b-2xl">
       <div class="flex items-baseline space-x-2 mb-2">
          <span class="text-3xl font-black text-red-600">${{ product?.price }}</span>
          <span class="text-sm text-gray-400 line-through">${{ product?.originalPrice }}</span>
          <span class="bg-red-100 text-red-600 text-xs font-bold px-1.5 py-0.5 rounded">-{{ product?.discount }}%</span>
       </div>
       <h1 class="text-lg font-bold text-gray-900 leading-snug mb-2">{{ product?.name }}</h1>
       <div class="flex items-center justify-between text-sm text-gray-500 border-t border-gray-50 pt-3 mt-1">
          <div class="flex items-center">
             <svg class="w-4 h-4 text-yellow-400 fill-current mr-1" viewBox="0 0 20 20"><path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"/></svg>
             <span class="font-bold text-black">{{ product?.rating }}</span>
             <span class="mx-1">·</span>
             <span>{{ Math.floor(product?.rating * 500) }} Reviews</span>
          </div>
          <span>{{ product?.sold }} Sold</span>
       </div>
    </div>

    <!-- Options & Shipping -->
    <div class="space-y-2">
       <!-- SKU Selection -->
       <div class="bg-white p-4 shadow-sm cursor-pointer group hover:bg-gray-50 transition-colors">
          <div class="flex justify-between items-center mb-2">
             <h3 class="font-bold text-gray-900">Select Options</h3>
             <span class="text-xs text-gray-500">{{ signatureStore.selected_sku_id ? 'Selected' : 'Please select' }}</span>
          </div>
          <div class="flex space-x-3 overflow-x-auto no-scrollbar pb-2">
             <div 
               id="sku-option-1" 
               class="flex-shrink-0 w-16 h-16 border rounded-md relative overflow-hidden cursor-pointer"
               :class="signatureStore.selected_sku_id ? 'border-red-600 ring-1 ring-red-600' : 'border-gray-200 hover:border-gray-400'"
               @click="handleSelectSKU"
             >
                <img :src="product?.image" class="w-full h-full object-cover" />
             </div>
             <div class="flex-shrink-0 w-16 h-16 border border-gray-200 rounded-md bg-gray-100 flex items-center justify-center text-xs text-gray-400">
                +3 more
             </div>
          </div>
          
          <!-- Quantity -->
          <div class="mt-4 flex items-center justify-between">
             <span class="text-sm font-medium text-gray-700">Quantity</span>
             <div class="flex items-center border border-gray-300 rounded-md">
                <button class="px-3 py-1 text-gray-600 hover:bg-gray-100 border-r border-gray-300" @click="quantity > 1 ? quantity-- : null">-</button>
                <input 
                  id="quantity-input" 
                  type="number" 
                  class="w-12 text-center py-1 text-sm font-bold focus:outline-none"
                  v-model="quantity"
                  @input="handleQuantityInput"
                />
                <button class="px-3 py-1 text-gray-600 hover:bg-gray-100 border-l border-gray-300" @click="quantity++">+</button>
             </div>
          </div>
       </div>

       <!-- Shipping -->
       <div class="bg-white p-4 shadow-sm">
          <div class="flex justify-between items-start mb-2">
             <h3 class="font-bold text-gray-900">Shipping</h3>
             <div id="ship-to-dropdown" class="relative group">
                <button class="text-xs font-bold text-blue-600 flex items-center">
                   To: {{ signatureStore.ship_to_country }}
                   <svg class="w-3 h-3 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
                </button>
                <!-- Dropdown -->
                <div class="absolute right-0 mt-1 w-32 bg-white border border-gray-100 shadow-lg rounded-md hidden group-hover:block z-20">
                   <div id="ship-to-us" class="px-3 py-2 hover:bg-gray-50 cursor-pointer text-sm" @click="handleShipTo('US')">🇺🇸 United States</div>
                   <div id="ship-to-uk" class="px-3 py-2 hover:bg-gray-50 cursor-pointer text-sm" @click="handleShipTo('UK')">🇬🇧 UK</div>
                   <div id="ship-to-cn" class="px-3 py-2 hover:bg-gray-50 cursor-pointer text-sm" @click="handleShipTo('CN')">🇨🇳 China</div>
                </div>
             </div>
          </div>
          <p class="text-sm text-gray-600 mb-1">Free Shipping</p>
          <p class="text-xs text-gray-400">Estimated delivery: Oct 24 - Nov 12</p>
       </div>

       <!-- Reviews Link -->
       <div 
         id="tab-reviews"
         class="bg-white p-4 shadow-sm flex justify-between items-center cursor-pointer hover:bg-gray-50"
         @click="handleGoReviews"
       >
          <div class="flex flex-col">
             <h3 class="font-bold text-gray-900">Reviews (234)</h3>
             <div class="flex items-center mt-1">
                <div class="flex text-yellow-400 text-xs">★★★★★</div>
                <span class="text-xs text-gray-500 ml-2">4.8/5</span>
             </div>
          </div>
          <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path></svg>
       </div>
       
       <!-- Contact Seller -->
       <div 
         id="contact-seller-button"
         class="bg-white p-4 shadow-sm flex justify-between items-center cursor-pointer hover:bg-gray-50 mb-20"
         @click="handleContactSeller"
       >
          <div class="flex items-center space-x-3">
             <div class="w-8 h-8 bg-orange-100 rounded-full flex items-center justify-center text-orange-600">
                💬
             </div>
             <span class="font-medium text-gray-900">Contact Seller</span>
          </div>
          <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path></svg>
       </div>
    </div>

    <!-- Bottom Actions -->
    <div class="fixed bottom-0 left-0 w-full bg-white border-t border-gray-100 p-2 flex items-center space-x-2 z-40 shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.05)]">
       <div class="flex flex-col items-center justify-center w-16 px-1 text-gray-500">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"></path></svg>
          <span class="text-[10px]">Store</span>
       </div>
       <div class="flex flex-col items-center justify-center w-16 px-1 text-gray-500 relative">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z"></path></svg>
          <span class="text-[10px]">Cart</span>
       </div>
       
       <button 
         id="add-to-cart-button"
         class="flex-1 bg-gradient-to-r from-orange-400 to-red-500 text-white font-bold py-2.5 rounded-full shadow-md active:scale-95 transition-transform disabled:opacity-50"
         :disabled="!canBuy"
         @click="handleAddToCart"
       >
         Add to Cart
       </button>
       <button 
         id="buy-now-button"
         class="flex-1 bg-gradient-to-r from-red-600 to-red-700 text-white font-bold py-2.5 rounded-full shadow-md active:scale-95 transition-transform disabled:opacity-50"
         :disabled="!canBuy"
         @click="handleBuyNow"
       >
         Buy Now
       </button>
    </div>

  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'PRODUCT_DETAIL',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()
    const quantity = ref(1)
    
    const product = computed(() => {
       return dataStore.products.find(p => p.id === signatureStore.selected_item_id) || {}
    })

    const canBuy = computed(() => {
       return signatureStore.selected_sku_id && quantity.value > 0
    })

    const handleBackList = async () => {
       signatureStore.currentPageId = 'PRODUCT_LIST'
       await router.push({ name: 'PRODUCT_LIST' })
    }

    const handleSelectSKU = () => {
       signatureStore.selected_sku_id = 'sku_1' // Simulate SKU selection
    }

    const handleQuantityInput = (e) => {
       signatureStore.quantity = parseInt(e.target.value) || 1
       quantity.value = signatureStore.quantity
    }

    const handleShipTo = (country) => {
       signatureStore.ship_to_country = country
    }

    const handleGoReviews = async () => {
       signatureStore.currentPageId = 'PRODUCT_REVIEWS'
       await router.push({ name: 'PRODUCT_REVIEWS' })
    }

    const handleContactSeller = async () => {
       signatureStore.currentPageId = 'CONTACT_SELLER_FORM'
       await router.push({ name: 'CONTACT_SELLER_FORM' })
    }

    const handleAddToCart = async () => {
       signatureStore.quantity = quantity.value
       signatureStore.buy_option = 'cart'
       signatureStore.currentPageId = 'ADD_TO_CART_CONFIRM'
       await router.push({ name: 'ADD_TO_CART_CONFIRM' })
    }

    const handleBuyNow = async () => {
       signatureStore.quantity = quantity.value
       signatureStore.buy_option = 'buynow'
       signatureStore.currentPageId = 'BUY_NOW_CHECKOUT'
       await router.push({ name: 'BUY_NOW_CHECKOUT' })
    }

    // Initialize quantity from store if set
    onMounted(() => {
       if(signatureStore.quantity) quantity.value = signatureStore.quantity
    })

    return {
       signatureStore,
       product,
       quantity,
       canBuy,
       handleBackList,
       handleSelectSKU,
       handleQuantityInput,
       handleShipTo,
       handleGoReviews,
       handleContactSeller,
       handleAddToCart,
       handleBuyNow
    }
  }
}
</script>