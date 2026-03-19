<template>
  <div class="min-h-screen bg-gray-50 font-sans">
    <!-- Header -->
    <header class="bg-white shadow-sm px-4 py-3 sticky top-0 z-20 flex items-center space-x-4">
      <button 
        id="account-back-home" 
        class="p-2 hover:bg-gray-100 rounded-full transition-colors"
        @click="handleBackHome"
      >
        <svg class="w-6 h-6 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"></path></svg>
      </button>
      <h1 class="text-lg font-bold text-gray-800">My Account</h1>
    </header>

    <main class="container mx-auto px-4 py-6 max-w-2xl">
      <!-- Profile Summary -->
      <div class="bg-white rounded-xl shadow-sm p-6 mb-6 flex items-center space-x-4">
        <div class="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center text-2xl">
          👤
        </div>
        <div>
          <h2 class="text-xl font-bold text-gray-900">John Doe</h2>
          <p class="text-sm text-gray-500">Member since 2024</p>
          <div class="flex items-center mt-1 space-x-2">
             <span class="bg-yellow-100 text-yellow-800 text-xs font-bold px-2 py-0.5 rounded">Gold Member</span>
          </div>
        </div>
      </div>

      <!-- Menu Grid -->
      <div class="grid grid-cols-1 gap-4">
        <!-- Orders -->
        <div 
          id="account-orders-link" 
          class="bg-white p-4 rounded-xl shadow-sm flex items-center justify-between cursor-pointer hover:bg-gray-50 transition-colors"
          @click="handleGoOrders"
        >
          <div class="flex items-center space-x-3">
            <div class="bg-blue-100 p-2 rounded-lg text-blue-600">
              <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 11V7a4 4 0 00-8 0v4M5 9h14l1 12H4L5 9z"></path></svg>
            </div>
            <span class="font-medium text-gray-700">My Orders</span>
          </div>
          <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path></svg>
        </div>

        <!-- Payment Methods -->
        <div 
          id="account-payments-link" 
          class="bg-white p-4 rounded-xl shadow-sm flex items-center justify-between cursor-pointer hover:bg-gray-50 transition-colors"
          @click="handleGoPayments"
        >
          <div class="flex items-center space-x-3">
            <div class="bg-green-100 p-2 rounded-lg text-green-600">
              <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z"></path></svg>
            </div>
            <span class="font-medium text-gray-700">Payment Methods</span>
          </div>
          <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path></svg>
        </div>

        <!-- Addresses -->
        <div 
          id="account-addresses-link" 
          class="bg-white p-4 rounded-xl shadow-sm flex items-center justify-between cursor-pointer hover:bg-gray-50 transition-colors"
          @click="handleGoAddresses"
        >
          <div class="flex items-center space-x-3">
            <div class="bg-orange-100 p-2 rounded-lg text-orange-600">
              <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
            </div>
            <span class="font-medium text-gray-700">Shipping Addresses</span>
          </div>
          <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path></svg>
        </div>

        <!-- Message Center (Mock) -->
        <div class="bg-white p-4 rounded-xl shadow-sm flex items-center justify-between cursor-pointer hover:bg-gray-50 transition-colors opacity-60">
          <div class="flex items-center space-x-3">
            <div class="bg-purple-100 p-2 rounded-lg text-purple-600">
              <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"></path></svg>
            </div>
            <span class="font-medium text-gray-700">Message Center</span>
          </div>
          <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path></svg>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'ACCOUNT_OVERVIEW',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const handleBackHome = async () => {
      signatureStore.currentPageId = 'HOME'
      await router.push({ name: 'HOME' })
    }

    const handleGoOrders = async () => {
      signatureStore.currentPageId = 'ORDERS_LIST'
      await router.push({ name: 'ORDERS_LIST' })
    }

    const handleGoPayments = async () => {
      signatureStore.currentPageId = 'PAYMENT_METHODS'
      await router.push({ name: 'PAYMENT_METHODS' })
    }

    const handleGoAddresses = async () => {
      signatureStore.currentPageId = 'ADDRESS_BOOK'
      await router.push({ name: 'ADDRESS_BOOK' })
    }

    return {
      handleBackHome,
      handleGoOrders,
      handleGoPayments,
      handleGoAddresses
    }
  }
}
</script>