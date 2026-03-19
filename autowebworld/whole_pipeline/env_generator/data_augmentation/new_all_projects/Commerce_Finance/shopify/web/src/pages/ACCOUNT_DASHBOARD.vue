<template>
  <div class="min-h-screen bg-gray-50 font-sans">
    <nav class="bg-white border-b border-gray-200">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <span class="text-xl font-bold text-[#008060]">My Account</span>
            <span 
                id="account-back-home" 
                @click="goHome"
                class="text-gray-500 hover:text-[#008060] cursor-pointer text-sm font-medium"
            >
                Log out
            </span>
        </div>
    </nav>

    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
            <!-- Order History Card -->
            <div class="bg-white p-8 rounded-xl shadow-sm border border-gray-100 hover:shadow-md transition-shadow">
                <div class="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-6 text-blue-600">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 11V7a4 4 0 00-8 0v4M5 9h14l1 12H4L5 9z" />
                    </svg>
                </div>
                <h2 class="text-xl font-bold text-gray-900 mb-2">Order History</h2>
                <p class="text-gray-500 mb-6">View and track your recent orders.</p>
                <button 
                    id="account-orders-link" 
                    @click="goToOrders"
                    class="text-[#008060] font-bold hover:underline flex items-center"
                >
                    View Orders 
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 ml-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 8l4 4m0 0l-4 4m4-4H3" />
                    </svg>
                </button>
            </div>

            <!-- Addresses Card -->
            <div class="bg-white p-8 rounded-xl shadow-sm border border-gray-100 hover:shadow-md transition-shadow">
                <div class="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mb-6 text-purple-600">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                    </svg>
                </div>
                <h2 class="text-xl font-bold text-gray-900 mb-2">Addresses</h2>
                <p class="text-gray-500 mb-6">Manage your shipping and billing addresses.</p>
                <button 
                    id="account-addresses-link" 
                    @click="goToAddresses"
                    class="text-[#008060] font-bold hover:underline flex items-center"
                >
                    View Addresses
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 ml-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 8l4 4m0 0l-4 4m4-4H3" />
                    </svg>
                </button>
            </div>
        </div>
    </main>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'ACCOUNT_DASHBOARD',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const goHome = async () => {
        signatureStore.currentPageId = 'HOME'
        await router.push({ name: 'HOME' })
    }

    const goToOrders = async () => {
        signatureStore.currentPageId = 'ORDER_HISTORY'
        await router.push({ name: 'ORDER_HISTORY' })
    }

    const goToAddresses = async () => {
        signatureStore.currentPageId = 'ACCOUNT_ADDRESSES'
        await router.push({ name: 'ACCOUNT_ADDRESSES' })
    }

    return {
        goHome,
        goToOrders,
        goToAddresses
    }
  }
}
</script>