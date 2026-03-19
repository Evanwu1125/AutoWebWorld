<template>
  <div class="min-h-screen bg-[#F6F6F6]">
    <!-- Header -->
    <header class="bg-white shadow-sm sticky top-0 z-20">
      <div class="container mx-auto px-4 py-4 flex items-center justify-between">
        <div class="flex items-center gap-4">
          <button id="back-home" @click="goHome" class="text-gray-500 hover:text-[#E1251B]">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"></path></svg>
          </button>
          <h1 class="text-xl font-bold">My Orders</h1>
        </div>
        <button id="go-cart" @click="goCart" class="text-gray-600 hover:text-[#E1251B] relative">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z"></path></svg>
        </button>
      </div>
    </header>

    <main class="container mx-auto px-4 py-6 max-w-4xl">
      <div id="orders-list" class="space-y-4">
        <div 
          v-for="order in orders" 
          :key="order.id"
          class="bg-white rounded-xl shadow-sm overflow-hidden cursor-pointer border border-transparent hover:border-[#E1251B] transition-all"
          :class="[getItemClass(order), `data-id-${order.id}`]"
          @click="openOrder(order)"
        >
          <div class="p-4 border-b bg-gray-50 flex justify-between items-center text-sm text-gray-500">
            <div class="flex items-center gap-4">
              <span class="font-medium text-gray-900">Order #{{ order.id.toUpperCase() }}</span>
              <span>{{ order.date }}</span>
            </div>
            <span 
              class="px-2 py-1 rounded text-xs font-bold"
              :class="{
                'bg-green-100 text-green-700': order.status === 'Delivered',
                'bg-blue-100 text-blue-700': order.status === 'Shipped',
                'bg-yellow-100 text-yellow-700': order.status === 'Processing',
                'bg-gray-100 text-gray-700': order.status === 'Cancelled'
              }"
            >
              {{ order.status.toUpperCase() }}
            </span>
          </div>
          
          <div class="p-4">
            <div class="flex gap-4 overflow-x-auto pb-2">
              <img v-for="(item, idx) in order.items" :key="idx" :src="item.image" class="w-16 h-16 object-cover rounded border border-gray-200" />
            </div>
            <div class="mt-4 flex justify-between items-center border-t pt-3">
              <div class="text-sm text-gray-500">{{ order.items.length }} items</div>
              <div class="text-lg font-bold text-[#E1251B]">${{ order.total }}</div>
            </div>
          </div>
        </div>
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
  name: 'ORDERS_LIST',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const orders = computed(() => dataStore.orders);
    const viewportAnchor = computed(() => signatureStore.orders_list_viewport_anchor_id);

    const getItemClass = (item) => {
      return 'row-visible';
    };

    const goHome = async () => {
      signatureStore.currentPageId = 'HOME';
      await router.push({ name: 'HOME' });
    };

    const goCart = async () => {
      signatureStore.currentPageId = 'CART';
      await router.push({ name: 'CART' });
    };

    const openOrder = async (order) => {
      signatureStore.orders_selected_item_id = order.id;
      signatureStore.orders_list_viewport_anchor_id = null;
      signatureStore.currentPageId = 'ORDER_DETAIL';
      await router.push({ name: 'ORDER_DETAIL', params: { id: order.id } });
    };

    return {
      orders,
      getItemClass,
      goHome,
      goCart,
      openOrder
    };
  }
}
</script>