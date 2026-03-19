<template>
  <div class="min-h-screen bg-white pb-20">
    <!-- Header -->
    <header class="border-b sticky top-0 bg-white z-20">
      <div class="container mx-auto px-4 py-3 flex items-center justify-between">
        <button id="back-category" @click="goBack" class="text-gray-600 hover:text-[#E1251B] flex items-center gap-2 font-medium">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
          Back
        </button>
        
        <nav class="flex gap-8 text-sm font-bold text-gray-800">
          <button class="text-[#E1251B] border-b-2 border-[#E1251B] pb-3 -mb-3.5">Product</button>
          <button id="tab-reviews" @click="goToReviews" class="hover:text-[#E1251B] pb-3 -mb-3.5 transition-colors">Reviews</button>
          <button class="hover:text-[#E1251B] pb-3 -mb-3.5 transition-colors">Details</button>
        </nav>
        
        <div class="w-20"></div> <!-- Spacer -->
      </div>
    </header>

    <main class="container mx-auto px-4 py-8" v-if="product">
      <div class="flex flex-col md:flex-row gap-12">
        <!-- Gallery -->
        <div class="w-full md:w-2/5">
          <div class="aspect-square bg-gray-50 rounded-xl overflow-hidden mb-4 border border-gray-100">
            <img :src="product.image" :alt="product.name" class="w-full h-full object-contain p-4" />
          </div>
          <div class="grid grid-cols-5 gap-2">
            <div class="aspect-square border-2 border-[#E1251B] rounded cursor-pointer p-1">
              <img :src="product.image" class="w-full h-full object-contain" />
            </div>
            <!-- Placeholders -->
            <div v-for="i in 4" :key="i" class="aspect-square border border-gray-200 rounded cursor-pointer hover:border-gray-400 bg-gray-50"></div>
          </div>
        </div>

        <!-- Info -->
        <div class="flex-1">
          <h1 class="text-2xl font-bold text-gray-900 mb-4 leading-snug">
            <span v-if="product.tags?.includes('Self-Operated')" class="bg-[#E1251B] text-white text-sm px-1.5 py-0.5 rounded mr-2 align-middle font-normal">JD</span>
            {{ product.name }}
          </h1>

          <div class="bg-[#f3f3f3] p-4 rounded-lg mb-6">
            <div class="flex items-baseline gap-2 mb-2">
              <span class="text-sm text-gray-500">JD Price</span>
              <span class="text-[#E1251B] text-3xl font-bold">${{ product.price }}</span>
              <span v-if="product.tags?.includes('Flash Sale')" class="bg-[#E1251B] text-white text-xs px-2 py-1 rounded ml-2">Flash Sale Ends in 05:23:11</span>
            </div>
          </div>

          <!-- SKU Selection -->
          <div class="mb-6">
            <h3 class="text-sm font-medium text-gray-500 mb-3">Select Option</h3>
            <div class="flex flex-wrap gap-3">
              <div 
                id="sku-option-1" 
                @click="selectSku('sku_1')" 
                class="px-4 py-2 border rounded cursor-pointer transition-all"
                :class="selectedSku === 'sku_1' ? 'border-[#E1251B] text-[#E1251B] bg-red-50 ring-1 ring-[#E1251B]' : 'border-gray-300 hover:border-gray-400'"
              >
                Standard Edition
              </div>
              <div class="px-4 py-2 border border-gray-300 rounded text-gray-400 cursor-not-allowed bg-gray-50">
                Pro Edition (Out of Stock)
              </div>
            </div>
          </div>

          <!-- Quantity -->
          <div class="mb-8">
            <h3 class="text-sm font-medium text-gray-500 mb-3">Quantity</h3>
            <div class="flex items-center border border-gray-300 rounded w-32 overflow-hidden">
              <button class="w-10 h-10 bg-gray-50 hover:bg-gray-100 border-r border-gray-300 flex items-center justify-center text-gray-600" @click="quantity = Math.max(1, quantity - 1)">-</button>
              <input 
                id="quantity-input" 
                type="number" 
                v-model.number="quantity" 
                class="flex-1 w-full text-center outline-none h-10"
              />
              <button class="w-10 h-10 bg-gray-50 hover:bg-gray-100 border-l border-gray-300 flex items-center justify-center text-gray-600" @click="quantity++">+</button>
            </div>
          </div>

          <!-- Actions -->
          <div class="flex gap-4">
            <button 
              id="btn-add-to-cart" 
              @click="addToCart" 
              class="flex-1 bg-[#FFD8D8] text-[#E1251B] font-bold py-4 rounded-lg hover:bg-[#ffcfcf] transition-colors border border-[#E1251B]"
            >
              Add to Cart
            </button>
            <button 
              id="btn-buy-now" 
              @click="buyNow" 
              class="flex-1 bg-gradient-to-r from-[#ff4142] to-[#ff0000] text-white font-bold py-4 rounded-lg shadow-lg hover:shadow-xl hover:from-[#ff2a2b] hover:to-[#e60000] transition-all"
            >
              Buy Now
            </button>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';

export default {
  name: 'PRODUCT_DETAIL',
  setup() {
    const route = useRoute();
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const product = ref(null);
    const quantity = ref(1);
    const selectedSku = computed(() => signatureStore.product_selected_sku_id);

    onMounted(() => {
      const id = route.params.id || signatureStore.electronics_selected_item_id || signatureStore.supermarket_selected_item_id || signatureStore.search_selected_item_id;
      
      const all = [...dataStore.electronics, ...dataStore.supermarket];
      product.value = all.find(p => p.id === id) || all[0];
      
      // Set initial quantity from signature if needed, mostly effect driven
    });

    const goBack = async () => {
      // Determine where to back based on product type or history
      if (product.value?.category) {
        signatureStore.currentPageId = 'CATEGORY_SUPERMARKET';
        await router.push({ name: 'CATEGORY_SUPERMARKET' });
      } else {
        signatureStore.currentPageId = 'CATEGORY_ELECTRONICS';
        await router.push({ name: 'CATEGORY_ELECTRONICS' });
      }
    };

    const goToReviews = async () => {
      signatureStore.currentPageId = 'PRODUCT_REVIEWS';
      await router.push({ name: 'PRODUCT_REVIEWS', params: { id: product.value.id } });
    };

    const selectSku = (skuId) => {
      signatureStore.product_selected_sku_id = skuId;
    };

    const addToCart = async () => {
      // Set quantity effect
      signatureStore.product_quantity = quantity.value;
      
      // Logic to add to cart store
      if (product.value) {
        dataStore.cart.push({
          id: 'c' + Date.now(),
          productId: product.value.id,
          name: product.value.name,
          price: product.value.price,
          quantity: quantity.value,
          image: product.value.image
        });
        signatureStore.cart_has_items = true;
      }

      signatureStore.currentPageId = 'CART';
      await router.push({ name: 'CART' });
    };

    const buyNow = async () => {
      signatureStore.product_quantity = quantity.value;
      signatureStore.currentPageId = 'CHECKOUT_BUY_NOW_CONFIRM';
      await router.push({ name: 'CHECKOUT_BUY_NOW_CONFIRM' });
    };

    return {
      product,
      quantity,
      selectedSku,
      goBack,
      goToReviews,
      selectSku,
      addToCart,
      buyNow
    };
  }
}
</script>