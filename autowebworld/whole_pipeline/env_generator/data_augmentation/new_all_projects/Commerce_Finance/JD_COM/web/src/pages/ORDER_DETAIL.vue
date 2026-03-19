<template>
  <div class="min-h-screen bg-[#F6F6F6] pb-20">
    <!-- Header -->
    <header class="bg-white shadow-sm sticky top-0 z-20">
      <div class="container mx-auto px-4 py-4 flex items-center gap-4">
        <button id="back-orders" @click="goBack" class="text-gray-600 hover:text-[#E1251B] flex items-center gap-1">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
          Back to Orders
        </button>
        <h1 class="text-xl font-bold">Order Details</h1>
      </div>
    </header>

    <main class="container mx-auto px-4 py-6 max-w-3xl" v-if="order">
      <!-- Status Card -->
      <div class="bg-gradient-to-r from-[#E1251B] to-[#ff6b6b] rounded-xl shadow-lg p-6 text-white mb-6">
        <h2 class="text-2xl font-bold mb-1">{{ order.status }}</h2>
        <p class="opacity-90">Order placed on {{ order.date }}</p>
      </div>

      <!-- Items -->
      <div class="bg-white rounded-xl shadow-sm p-6 mb-6">
        <h3 class="font-bold text-gray-900 mb-4 border-b pb-2">Items Purchased</h3>
        <div class="space-y-4">
          <div v-for="(item, idx) in order.items" :key="idx" class="flex gap-4">
            <img :src="item.image" class="w-20 h-20 object-cover rounded border" />
            <div class="flex-1">
              <div class="font-medium text-gray-900">{{ item.name }}</div>
              <div class="text-[#E1251B] font-bold mt-1">${{ item.price }}</div>
            </div>
            <button 
              id="btn-apply-after-sale"
              @click="applyAfterSale"
              class="px-4 py-2 border border-gray-300 rounded text-sm hover:border-[#E1251B] hover:text-[#E1251B] transition-colors h-fit"
            >
              Apply for Service
            </button>
          </div>
        </div>
      </div>

      <!-- Info -->
      <div class="bg-white rounded-xl shadow-sm p-6">
        <h3 class="font-bold text-gray-900 mb-4 border-b pb-2">Order Info</h3>
        <div class="space-y-2 text-sm">
          <div class="flex justify-between">
            <span class="text-gray-500">Order ID</span>
            <span class="text-gray-900">{{ order.id }}</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-500">Total Amount</span>
            <span class="text-[#E1251B] font-bold">${{ order.total }}</span>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { ref, onMounted } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';

export default {
  name: 'ORDER_DETAIL',
  setup() {
    const route = useRoute();
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const order = ref(null);

    onMounted(() => {
      const id = route.params.id || signatureStore.orders_selected_item_id;
      order.value = dataStore.orders.find(o => o.id === id) || dataStore.orders[0];
    });

    const goBack = async () => {
      signatureStore.currentPageId = 'ORDERS_LIST';
      await router.push({ name: 'ORDERS_LIST' });
    };

    const applyAfterSale = async () => {
      signatureStore.order_can_apply_service = true;
      signatureStore.currentPageId = 'AFTER_SALE_APPLY';
      await router.push({ name: 'AFTER_SALE_APPLY' });
    };

    return {
      order,
      goBack,
      applyAfterSale
    };
  }
}
</script>