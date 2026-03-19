<template>
  <div class="grocery-detail-page min-h-screen bg-white flex flex-col">
    <header class="bg-white border-b sticky top-0 z-30">
      <div class="max-w-7xl mx-auto px-4 py-4 flex items-center gap-4">
        <div 
          id="grocery-breadcrumb-category" 
          @click="handleBackToCategory"
          class="cursor-pointer text-gray-500 hover:text-[#2A8703] flex items-center gap-1 text-sm font-medium"
        >
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" /></svg>
          Back to Groceries
        </div>
      </div>
    </header>

    <main v-if="product" class="flex-1 max-w-7xl mx-auto w-full p-4 md:p-8">
      <div class="grid md:grid-cols-2 gap-12">
        <!-- Image -->
        <div class="aspect-square bg-gray-50 rounded-2xl overflow-hidden flex items-center justify-center p-8">
           <img :src="product.image" :alt="product.name" class="max-w-full max-h-full object-contain mix-blend-multiply" />
           <div v-if="product.type === 'organic'" class="absolute top-4 left-4 bg-[#2A8703] text-white font-bold px-3 py-1 rounded-full uppercase tracking-wider text-sm shadow-md">
             Organic
           </div>
        </div>

        <!-- Info -->
        <div>
          <div class="mb-6 border-b pb-6">
             <h1 class="text-3xl font-bold text-gray-900 mb-2">{{ product.name }}</h1>
             <div class="text-4xl font-bold text-gray-900">${{ product.price.toFixed(2) }} <span class="text-lg text-gray-500 font-normal">/ {{ product.unit }}</span></div>
          </div>

          <!-- Quantity -->
          <div class="mb-8">
            <label class="block text-sm font-semibold text-gray-700 mb-2">Quantity</label>
            <div class="flex items-center gap-4">
              <input 
                id="grocery-quantity-input"
                type="number" 
                v-model.number="quantity"
                @input="handleSetQuantity"
                min="1"
                class="w-24 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-[#2A8703] outline-none font-medium text-center text-lg"
              />
              <div class="text-gray-500 text-sm">
                Total: <span class="font-bold text-gray-900">${{ (product.price * quantity).toFixed(2) }}</span>
              </div>
            </div>
          </div>

          <!-- Add to Cart -->
          <button 
            id="grocery-add-to-cart-button"
            @click="handleAddToCart"
            class="w-full md:w-auto bg-[#2A8703] text-white font-bold py-4 px-12 rounded-full shadow-lg hover:bg-[#237002] transition-all transform hover:-translate-y-0.5 flex items-center justify-center gap-2"
          >
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z" /></svg>
            Add to Order
          </button>
          
          <div class="mt-8 p-4 bg-green-50 rounded-lg border border-green-100 text-sm text-green-800 flex items-start gap-3">
             <div class="mt-0.5">🌱</div>
             <div>
               <p class="font-bold mb-1">Freshness Guarantee</p>
               <p>We pick the freshest items for you, or your money back. 100% satisfaction guaranteed.</p>
             </div>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'GROCERY_PRODUCT_DETAIL',
  setup() {
    const route = useRoute()
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const productId = route.params.id || store.selected_product_id
    const product = computed(() => dataStore.groceries.find(p => p.id === productId))
    
    const quantity = ref(1)

    const handleSetQuantity = () => {
      // FSM: ACT_GROCERY_DETAIL_SET_QUANTITY
      store.grocery_selected_quantity = quantity.value
    }

    const handleAddToCart = async () => {
      // FSM: ACT_GROCERY_DETAIL_ADD_TO_CART
      store.grocery_cart_items.push({ id: product.value.id, qty: quantity.value })
      
      store.currentPageId = 'GROCERY_CART'
      await router.push({ name: 'GROCERY_CART' })
    }

    const handleBackToCategory = async () => {
      // FSM: ACT_GROCERY_DETAIL_BACK_TO_CATEGORY
      store.currentPageId = 'GROCERY_CATEGORY'
      await router.push({ name: 'GROCERY_CATEGORY' })
    }

    onMounted(() => {
      store.grocery_selected_quantity = 1
    })

    return {
      product,
      quantity,
      handleSetQuantity,
      handleAddToCart,
      handleBackToCategory
    }
  }
}
</script>