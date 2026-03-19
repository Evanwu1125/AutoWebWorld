<template>
  <div class="min-h-screen bg-[#F6F6F6] pb-20">
    <!-- Header -->
    <header class="bg-white shadow-sm sticky top-0 z-20">
      <div class="container mx-auto px-4 py-4 flex items-center justify-between">
        <div class="flex items-center gap-4">
          <button id="back-home" @click="goHome" class="text-gray-500 hover:text-[#E1251B]">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"></path></svg>
          </button>
          <h1 class="text-xl font-bold">Shopping Cart</h1>
        </div>
      </div>
    </header>

    <main class="container mx-auto px-4 py-6">
      <div v-if="cartItems.length > 0" class="flex flex-col lg:flex-row gap-6">
        <!-- Cart Items List -->
        <div class="flex-1 bg-white rounded-xl shadow-sm overflow-hidden">
          <div class="p-4 border-b bg-gray-50 flex justify-between text-sm font-medium text-gray-500">
            <span>Product</span>
            <span>Total</span>
          </div>
          
          <div v-for="item in cartItems" :key="item.id" class="p-4 border-b last:border-0 flex gap-4">
            <div class="w-24 h-24 border rounded bg-gray-50 p-2 flex-shrink-0">
              <img :src="item.image" :alt="item.name" class="w-full h-full object-contain" />
            </div>
            <div class="flex-1 flex flex-col justify-between">
              <div>
                <h3 class="font-medium text-gray-900 line-clamp-2 mb-1">{{ item.name }}</h3>
                <p class="text-sm text-gray-500">Stock: Available</p>
              </div>
              <div class="flex justify-between items-end">
                <div class="flex items-center border rounded overflow-hidden h-8">
                  <input 
                    type="number" 
                    v-model.number="item.quantity" 
                    @input="updateQuantity"
                    class="cart-item-qty-input w-16 text-center outline-none h-full text-sm"
                  />
                </div>
                <div class="font-bold text-[#E1251B]">${{ (item.price * item.quantity).toFixed(2) }}</div>
              </div>
            </div>
          </div>
        </div>

        <!-- Summary -->
        <div class="w-full lg:w-80">
          <div class="bg-white rounded-xl shadow-sm p-6 sticky top-24">
            <h3 class="text-lg font-bold mb-4">Order Summary</h3>
            <div class="flex justify-between mb-2 text-gray-600">
              <span>Subtotal</span>
              <span>${{ total.toFixed(2) }}</span>
            </div>
            <div class="flex justify-between mb-4 text-gray-600">
              <span>Shipping</span>
              <span>Free</span>
            </div>
            <div class="border-t pt-4 mb-6 flex justify-between items-end">
              <span class="font-bold text-gray-900">Total</span>
              <span class="text-2xl font-bold text-[#E1251B]">${{ total.toFixed(2) }}</span>
            </div>
            <button 
              id="btn-cart-checkout"
              @click="checkout"
              class="w-full bg-[#E1251B] hover:bg-[#c91f16] text-white font-bold py-3 rounded-lg shadow-lg shadow-red-200 transition-colors"
            >
              Proceed to Checkout
            </button>
            <button 
              id="continue-shopping"
              @click="goShopping"
              class="w-full mt-3 bg-white border border-gray-300 text-gray-700 font-medium py-3 rounded-lg hover:bg-gray-50 transition-colors"
            >
              Continue Shopping
            </button>
          </div>
        </div>
      </div>

      <!-- Empty State -->
      <div v-else class="text-center py-20">
        <div class="text-6xl mb-4">🛒</div>
        <h2 class="text-2xl font-bold text-gray-900 mb-2">Your cart is empty</h2>
        <p class="text-gray-500 mb-8">Looks like you haven't added anything to your cart yet.</p>
        <button 
          id="continue-shopping"
          @click="goShopping"
          class="bg-[#E1251B] text-white font-bold px-8 py-3 rounded-full shadow-lg hover:scale-105 transition-transform"
        >
          Start Shopping
        </button>
      </div>
    </main>
  </div>
</template>

<script>
import { computed } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';

export default {
  name: 'CART',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const cartItems = computed(() => dataStore.cart);
    
    const total = computed(() => {
      return cartItems.value.reduce((sum, item) => sum + (item.price * item.quantity), 0);
    });

    const updateQuantity = () => {
      signatureStore.cart_has_items = cartItems.value.length > 0;
    };

    const checkout = async () => {
      signatureStore.currentPageId = 'CHECKOUT_CART_CONFIRM';
      await router.push({ name: 'CHECKOUT_CART_CONFIRM' });
    };

    const goHome = async () => {
      signatureStore.currentPageId = 'HOME';
      await router.push({ name: 'HOME' });
    };

    const goShopping = async () => {
      signatureStore.currentPageId = 'CATEGORY_ELECTRONICS';
      await router.push({ name: 'CATEGORY_ELECTRONICS' });
    };

    return {
      cartItems,
      total,
      updateQuantity,
      checkout,
      goHome,
      goShopping
    };
  }
}
</script>