<template>
  <div class="min-h-screen bg-[#F6F6F6]">
    <!-- Header -->
    <header class="bg-[#E1251B] text-white sticky top-0 z-20 shadow-md">
      <div class="container mx-auto px-4 py-4 flex items-center justify-between">
        <h1 class="text-xl font-bold">User Center</h1>
        <button id="back-home" @click="goHome" class="bg-white/20 hover:bg-white/30 px-4 py-1 rounded-full text-sm transition-colors">
          Home
        </button>
      </div>
      
      <!-- Profile Summary -->
      <div class="container mx-auto px-4 pb-8 pt-4 flex items-center gap-4">
        <div class="w-16 h-16 bg-white rounded-full p-1">
          <img src="/images/Profile.jpg" class="w-full h-full rounded-full object-cover" />
        </div>
        <div>
          <div class="text-lg font-bold">John Doe</div>
          <div class="text-white/80 text-sm">Silver Member</div>
        </div>
      </div>
    </header>

    <main class="container mx-auto px-4 py-6 -mt-6 relative z-10">
      <div class="bg-white rounded-xl shadow-sm overflow-hidden mb-6">
        <div class="p-4 border-b font-bold text-gray-900">My Services</div>
        <div class="grid grid-cols-3 divide-x divide-gray-100">
          <div id="menu-orders" @click="goToOrders" class="p-6 flex flex-col items-center gap-2 cursor-pointer hover:bg-gray-50 transition-colors">
            <span class="text-2xl">📦</span>
            <span class="text-sm font-medium text-gray-700">My Orders</span>
          </div>
          <div id="menu-address" @click="goToAddress" class="p-6 flex flex-col items-center gap-2 cursor-pointer hover:bg-gray-50 transition-colors">
            <span class="text-2xl">📍</span>
            <span class="text-sm font-medium text-gray-700">Addresses</span>
          </div>
          <div id="menu-payment" @click="goToPayment" class="p-6 flex flex-col items-center gap-2 cursor-pointer hover:bg-gray-50 transition-colors">
            <span class="text-2xl">💳</span>
            <span class="text-sm font-medium text-gray-700">Payments</span>
          </div>
        </div>
      </div>

      <!-- Promo Banner -->
      <div class="bg-gradient-to-r from-purple-500 to-indigo-600 rounded-xl p-6 text-white shadow-lg mb-6">
        <h3 class="font-bold text-lg mb-1">Plus Membership</h3>
        <p class="text-white/90 text-sm mb-4">Get free shipping on all orders + 10x points.</p>
        <button class="bg-white text-purple-600 px-4 py-2 rounded-lg font-bold text-sm">Upgrade Now</button>
      </div>
    </main>
  </div>
</template>

<script>
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'USER_CENTER',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();

    const goHome = async () => {
      signatureStore.currentPageId = 'HOME';
      await router.push({ name: 'HOME' });
    };

    const goToOrders = async () => {
      signatureStore.currentPageId = 'ORDERS_LIST';
      await router.push({ name: 'ORDERS_LIST' });
    };

    const goToAddress = async () => {
      signatureStore.currentPageId = 'ADDRESS_BOOK';
      await router.push({ name: 'ADDRESS_BOOK' });
    };

    const goToPayment = async () => {
      signatureStore.currentPageId = 'PAYMENT_METHODS';
      await router.push({ name: 'PAYMENT_METHODS' });
    };

    return {
      goHome,
      goToOrders,
      goToAddress,
      goToPayment
    };
  }
}
</script>