<template>
  <div class="min-h-screen bg-slate-50 font-sans">
    <!-- Account Header -->
    <header class="bg-white shadow-sm border-b border-gray-200">
      <div class="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        <div id="logo-home" @click="goHome" class="text-2xl font-bold text-[#002D5C] cursor-pointer flex items-center gap-2">
           <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M21 16v-2l-8-5V3.5A1.5 1.5 0 0 0 11.5 2 1.5 1.5 0 0 0 10 3.5V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5l8 2.5z"/></svg>
           Skyscanner
        </div>
        <div class="flex items-center gap-4">
          <div class="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center text-blue-700 font-bold">JD</div>
          <button id="back-login" @click="logout" class="text-sm font-medium text-gray-500 hover:text-red-600 transition-colors">
            Sign Out
          </button>
        </div>
      </div>
    </header>

    <main class="max-w-7xl mx-auto px-6 py-12">
      <div class="flex flex-col md:flex-row gap-8">
        <!-- Sidebar -->
        <aside class="w-full md:w-64 shrink-0">
          <nav class="bg-white rounded-xl shadow-sm p-4 space-y-2">
            <div class="px-4 py-2 font-bold text-gray-900">My Account</div>
            <a href="#" class="block px-4 py-2 text-blue-600 bg-blue-50 rounded-lg font-medium">Overview</a>
            <a id="account-nav-price-alerts" @click="goToAlerts" class="block px-4 py-2 text-gray-600 hover:bg-gray-50 rounded-lg cursor-pointer transition-colors">Price Alerts</a>
            <a id="account-nav-flights" @click="goToFlights" class="block px-4 py-2 text-gray-600 hover:bg-gray-50 rounded-lg cursor-pointer transition-colors">Search Flights</a>
          </nav>
        </aside>

        <!-- Main Content -->
        <div class="flex-1 space-y-8">
          <div class="bg-white rounded-2xl shadow-sm p-8 border border-gray-100">
            <h1 class="text-2xl font-bold text-gray-900 mb-2">Welcome, John</h1>
            <p class="text-gray-500 mb-8">Manage your trips and account settings.</p>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
              <!-- Quick Actions -->
              <div @click="goToFlights" class="group cursor-pointer bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl p-6 text-white shadow-lg hover:shadow-xl transition-all transform hover:-translate-y-1">
                <div class="flex justify-between items-start mb-4">
                  <svg class="w-8 h-8 opacity-80" fill="currentColor" viewBox="0 0 24 24"><path d="M21 16v-2l-8-5V3.5A1.5 1.5 0 0 0 11.5 2 1.5 1.5 0 0 0 10 3.5V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5l8 2.5z"/></svg>
                  <span class="bg-white/20 px-3 py-1 rounded-full text-xs font-medium">Start Now</span>
                </div>
                <h3 class="text-xl font-bold mb-1">Book a Flight</h3>
                <p class="text-blue-100 text-sm">Find the best deals for your next adventure.</p>
              </div>

              <div @click="goToAlerts" class="group cursor-pointer bg-white border border-gray-200 rounded-xl p-6 hover:border-blue-300 hover:shadow-md transition-all">
                <div class="flex justify-between items-start mb-4">
                   <svg class="w-8 h-8 text-yellow-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"/></svg>
                </div>
                <h3 class="text-xl font-bold text-gray-900 mb-1 group-hover:text-blue-600 transition-colors">Price Alerts</h3>
                <p class="text-gray-500 text-sm">Track prices and get notified when they drop.</p>
              </div>
            </div>
          </div>
          
          <div class="bg-white rounded-2xl shadow-sm p-8 border border-gray-100">
             <h2 class="text-lg font-bold text-gray-900 mb-4">Recent Activity</h2>
             <div class="space-y-4">
               <div class="flex items-center gap-4 p-4 rounded-lg bg-gray-50">
                 <div class="w-2 h-2 rounded-full bg-green-500"></div>
                 <span class="text-gray-600 text-sm">Logged in successfully</span>
                 <span class="text-gray-400 text-xs ml-auto">Just now</span>
               </div>
             </div>
          </div>
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
    const store = useSignatureStore()

    const goHome = async () => {
      store.currentPageId = 'HOME'
      await router.push({ name: 'HOME' })
    }

    const goToFlights = async () => {
      store.currentPageId = 'FLIGHTS_SEARCH'
      await router.push({ name: 'FLIGHTS_SEARCH' })
    }

    const goToAlerts = async () => {
      store.currentPageId = 'PRICE_ALERTS_LIST'
      await router.push({ name: 'PRICE_ALERTS_LIST' })
    }

    const logout = async () => {
      store.currentPageId = 'LOGIN'
      await router.push({ name: 'LOGIN' })
    }

    return {
      goHome,
      goToFlights,
      goToAlerts,
      logout
    }
  }
}
</script>