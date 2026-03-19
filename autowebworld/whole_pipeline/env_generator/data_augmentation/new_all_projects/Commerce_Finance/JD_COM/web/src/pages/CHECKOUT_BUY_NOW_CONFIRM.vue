<template>
  <div class="min-h-screen bg-[#F6F6F6] pb-24">
    <!-- Header -->
    <header class="bg-white shadow-sm sticky top-0 z-20">
      <div class="container mx-auto px-4 py-4 flex items-center gap-4">
        <button id="back-product" @click="goBack" class="text-gray-600 hover:text-[#E1251B] flex items-center gap-1">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
          Back to Product
        </button>
        <h1 class="text-xl font-bold">Checkout (Buy Now)</h1>
      </div>
    </header>

    <main class="container mx-auto px-4 py-6 max-w-3xl">
      <!-- Address Section -->
      <section class="bg-white rounded-xl shadow-sm p-6 mb-6">
        <h2 class="text-lg font-bold mb-4 flex items-center gap-2">
          <span class="bg-[#E1251B] text-white w-6 h-6 rounded-full flex items-center justify-center text-sm">1</span>
          Shipping Address
        </h2>
        <div 
          class="address-item-default border-2 rounded-lg p-4 cursor-pointer transition-all hover:border-red-200 relative"
          :class="addressSelected ? 'border-[#E1251B] bg-red-50' : 'border-gray-200'"
          @click="selectAddress"
        >
          <div class="flex justify-between items-start mb-2">
            <span class="font-bold text-gray-900">John Doe</span>
            <span class="text-gray-500">138****8888</span>
          </div>
          <p class="text-gray-600 text-sm">123 Tech Park, Beijing, China</p>
          <div v-if="addressSelected" class="absolute top-2 right-2 text-[#E1251B]">
            <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path></svg>
          </div>
        </div>
      </section>

      <!-- Payment Section -->
      <section class="bg-white rounded-xl shadow-sm p-6 mb-6">
        <h2 class="text-lg font-bold mb-4 flex items-center gap-2">
          <span class="bg-[#E1251B] text-white w-6 h-6 rounded-full flex items-center justify-center text-sm">2</span>
          Payment Method
        </h2>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div 
            class="payment-option-cod border-2 rounded-lg p-4 cursor-pointer hover:border-red-200 text-center"
            :class="paymentSelected ? 'border-[#E1251B] bg-red-50' : 'border-gray-200'"
            @click="selectPayment"
          >
            <div class="text-2xl mb-2">💵</div>
            <div class="font-medium">Cash on Delivery</div>
          </div>
          <div class="border-2 border-gray-200 rounded-lg p-4 cursor-not-allowed opacity-50 text-center bg-gray-50">
            <div class="text-2xl mb-2">💳</div>
            <div class="font-medium">Online Payment</div>
          </div>
        </div>
      </section>

      <!-- Order Item Preview -->
      <section class="bg-white rounded-xl shadow-sm p-6 mb-6" v-if="product">
        <h2 class="text-lg font-bold mb-4">Item</h2>
        <div class="flex gap-4">
          <img :src="product.image" class="w-20 h-20 object-cover rounded border" />
          <div>
            <div class="font-medium text-lg">{{ product.name }}</div>
            <div class="text-gray-500 mt-1">Qty: {{ quantity }}</div>
            <div class="text-[#E1251B] font-bold text-xl mt-2">${{ product.price }}</div>
          </div>
        </div>
      </section>

      <!-- Footer Bar -->
      <div class="fixed bottom-0 left-0 right-0 bg-white border-t p-4 shadow-lg z-30">
        <div class="container mx-auto max-w-3xl flex justify-end items-center gap-6">
          <div class="text-right">
            <span class="text-gray-500 mr-2">Total:</span>
            <span class="text-3xl font-bold text-[#E1251B]">${{ total.toFixed(2) }}</span>
          </div>
          <button 
            id="btn-submit-order"
            @click="submitOrder"
            class="bg-[#E1251B] hover:bg-[#c91f16] text-white font-bold py-3 px-8 rounded-full shadow-lg shadow-red-200 disabled:opacity-50 disabled:cursor-not-allowed"
            :disabled="!canSubmit"
          >
            Submit Order
          </button>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';

export default {
  name: 'CHECKOUT_BUY_NOW_CONFIRM',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const product = ref(null);
    const quantity = computed(() => signatureStore.product_quantity || 1);

    const addressSelected = computed(() => signatureStore.checkout_bn_address_selected);
    const paymentSelected = computed(() => signatureStore.checkout_bn_payment_selected);
    
    onMounted(() => {
      const id = signatureStore.product_selected_item_id || signatureStore.electronics_selected_item_id || signatureStore.supermarket_selected_item_id || signatureStore.search_selected_item_id;
      const all = [...dataStore.electronics, ...dataStore.supermarket];
      product.value = all.find(p => p.id === id) || all[0];
    });

    const total = computed(() => {
      return product.value ? product.value.price * quantity.value : 0;
    });

    const canSubmit = computed(() => addressSelected.value && paymentSelected.value);

    const selectAddress = () => {
      signatureStore.checkout_bn_address_selected = true;
    };

    const selectPayment = () => {
      signatureStore.checkout_bn_payment_selected = true;
    };

    const submitOrder = async () => {
      signatureStore.currentPageId = 'CHECKOUT_BUY_NOW_SUCCESS';
      await router.push({ name: 'CHECKOUT_BUY_NOW_SUCCESS' });
    };

    const goBack = async () => {
      signatureStore.currentPageId = 'PRODUCT_DETAIL';
      await router.push({ name: 'PRODUCT_DETAIL' });
    };

    return {
      product,
      quantity,
      addressSelected,
      paymentSelected,
      total,
      canSubmit,
      selectAddress,
      selectPayment,
      submitOrder,
      goBack
    };
  }
}
</script>