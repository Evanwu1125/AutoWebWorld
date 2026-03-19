<template>
  <div class="min-h-screen bg-gray-50 flex flex-col font-sans">
    <!-- Top Navigation Bar -->
    <header class="sticky top-0 z-50 bg-white shadow-md">
      <div class="container mx-auto px-4 py-3 flex items-center justify-between">
        <!-- Logo -->
        <div class="flex items-center space-x-2">
          <h1 class="text-2xl font-black text-red-600 tracking-tighter">AliExpress</h1>
        </div>

        <!-- Search Bar (Visual only) -->
        <div class="hidden md:flex flex-1 mx-8 max-w-2xl">
          <div class="relative w-full">
            <input 
              type="text" 
              placeholder="Search for products, brands and more..." 
              class="w-full pl-4 pr-10 py-2 border-2 border-red-600 rounded-l-full rounded-r-full focus:outline-none"
            />
            <button class="absolute right-0 top-0 bottom-0 bg-red-600 text-white px-6 rounded-r-full hover:bg-red-700 transition-colors">
              Search
            </button>
          </div>
        </div>

        <!-- Right Actions -->
        <div class="flex items-center space-x-6">
          <!-- Hover Menu for Deals -->
          <div id="home-top-nav" class="relative group py-2">
            <button class="flex items-center space-x-1 text-gray-700 hover:text-red-600 font-medium">
              <span>Deals</span>
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
            </button>
            <!-- Dropdown -->
            <div class="absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 transform origin-top-right z-50 border border-gray-100">
              <div class="py-1">
                <a 
                  id="nav-flash-deals" 
                  href="#" 
                  class="block px-4 py-2 text-sm text-gray-700 hover:bg-red-50 hover:text-red-600"
                  @click.prevent="handleGoDeals"
                >
                  ⚡ Flash Deals
                </a>
                <a href="#" class="block px-4 py-2 text-sm text-gray-700 hover:bg-red-50 hover:text-red-600">
                  💎 Super Value
                </a>
              </div>
            </div>
          </div>

          <!-- Account Menu -->
          <div class="relative group">
            <div id="account-menu-toggle" class="flex items-center space-x-2 cursor-pointer text-gray-700 hover:text-red-600">
              <div class="bg-gray-100 p-2 rounded-full">
                <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path></svg>
              </div>
              <span class="hidden sm:block font-medium">Account</span>
            </div>
            <div class="absolute right-0 mt-2 w-56 bg-white rounded-lg shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 transform origin-top-right z-50 border border-gray-100 p-4">
              <p class="text-xs text-gray-500 mb-3">Welcome back!</p>
              <button 
                id="account-menu-login" 
                class="w-full bg-red-600 text-white font-bold py-2 rounded-md hover:bg-red-700 transition-colors mb-2"
                @click="handleGoAccount"
              >
                Sign In / Join
              </button>
              <div class="border-t border-gray-100 pt-2 mt-2 space-y-2">
                <a href="#" class="block text-sm text-gray-600 hover:text-red-600">My Orders</a>
                <a href="#" class="block text-sm text-gray-600 hover:text-red-600">Message Center</a>
              </div>
            </div>
          </div>

          <!-- Cart -->
          <div 
            id="nav-cart-icon" 
            class="relative cursor-pointer text-gray-700 hover:text-red-600 transition-colors"
            @click="handleGoCart"
          >
            <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z"></path></svg>
            <span class="absolute -top-2 -right-2 bg-red-600 text-white text-xs font-bold rounded-full h-5 w-5 flex items-center justify-center border-2 border-white">
              {{ cartCount }}
            </span>
          </div>
        </div>
      </div>
    </header>

    <main class="flex-grow">
      <!-- Hero Section -->
      <div class="bg-white mb-8">
        <div class="container mx-auto px-4 py-6 grid grid-cols-1 md:grid-cols-4 gap-4">
          <!-- Sidebar Categories (Visual) -->
          <div class="hidden md:block col-span-1 bg-gray-50 rounded-lg p-4 border border-gray-100 shadow-sm">
            <h3 class="font-bold text-gray-800 mb-4 px-2">Categories</h3>
            <ul class="space-y-2">
              <li v-for="cat in ['Women\'s Fashion', 'Men\'s Fashion', 'Phones & Comms', 'Computer & Office', 'Consumer Electronics', 'Jewelry & Watches']" :key="cat" class="px-2 py-2 hover:bg-white hover:shadow-sm rounded-md cursor-pointer text-sm text-gray-600 hover:text-red-600 transition-all">
                {{ cat }}
              </li>
            </ul>
          </div>
          
          <!-- Main Banner -->
          <div class="col-span-1 md:col-span-3 relative h-64 md:h-96 rounded-2xl overflow-hidden shadow-lg group cursor-pointer">
             <img 
               src="/images/SuperDeals.jpg" 
               alt="Super Deals Promo" 
               class="w-full h-full object-cover transition-transform duration-700 group-hover:scale-105"
             />
             <div class="absolute inset-0 bg-gradient-to-r from-black/60 to-transparent flex flex-col justify-center px-8 md:px-16">
                <h2 class="text-3xl md:text-5xl font-black text-white mb-4 leading-tight">
                  Global Shopping <br/> Festival
                </h2>
                <p class="text-white/90 text-lg mb-8 max-w-md">Millions of items at unbeatable prices. Free shipping on orders over $10.</p>
                <button 
                  id="home-hot-categories" 
                  class="w-fit bg-white text-red-600 font-bold py-3 px-8 rounded-full shadow-lg hover:bg-red-50 transition-colors transform hover:-translate-y-1"
                  @click="handleGoCategories"
                >
                  Shop Categories
                </button>
             </div>
          </div>
        </div>
      </div>

      <!-- Featured Sections -->
      <div class="container mx-auto px-4 pb-12 space-y-12">
        
        <!-- Flash Deals Teaser -->
        <section>
          <div class="flex items-center justify-between mb-6">
            <h2 class="text-2xl font-bold text-gray-900 flex items-center">
              <span class="bg-red-600 text-white p-1 rounded mr-2 text-lg">⚡</span>
              Flash Deals
            </h2>
            <div class="text-sm text-gray-500">Ends in 04:23:12</div>
          </div>
          <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
            <div v-for="i in 6" :key="i" class="bg-white rounded-lg shadow-sm hover:shadow-md transition-shadow p-3 cursor-pointer border border-transparent hover:border-red-100">
              <div class="aspect-square rounded-md bg-gray-100 mb-3 relative overflow-hidden">
                <img :src="`/images/product-thumb-${i}.jpg`" class="w-full h-full object-cover" alt="Product" />
                <div class="absolute top-0 left-0 bg-yellow-400 text-red-700 text-xs font-bold px-2 py-1 rounded-br-lg">-45%</div>
              </div>
              <div class="text-lg font-bold text-red-600 mb-1">$12.99</div>
              <div class="text-xs text-gray-400 line-through mb-2">$24.99</div>
              <div class="w-full bg-gray-200 rounded-full h-1.5 mb-1">
                <div class="bg-red-500 h-1.5 rounded-full" style="width: 75%"></div>
              </div>
              <div class="text-xs text-gray-500">75% Sold</div>
            </div>
          </div>
        </section>

        <!-- More to Love -->
        <section>
          <h2 class="text-2xl font-bold text-gray-900 mb-6 text-center">More to Love</h2>
          <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-6">
            <div v-for="i in 10" :key="`love-${i}`" class="bg-white rounded-xl shadow-sm hover:shadow-xl transition-all duration-300 overflow-hidden group">
              <div class="aspect-[4/5] relative overflow-hidden">
                <img :src="`/images/product-lifestyle-${i}.jpg`" class="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110" alt="Product" />
              </div>
              <div class="p-4">
                <h3 class="text-sm text-gray-700 font-medium line-clamp-2 mb-2 group-hover:text-red-600 transition-colors">Premium Wireless Headphones with Noise Cancellation</h3>
                <div class="flex items-baseline space-x-2 mb-1">
                   <span class="text-xl font-black text-red-600">$45.99</span>
                   <span class="text-xs text-gray-400">10k+ sold</span>
                </div>
                <div class="flex items-center space-x-1">
                  <span class="bg-green-100 text-green-700 text-[10px] font-bold px-1 rounded">Free Shipping</span>
                </div>
              </div>
            </div>
          </div>
        </section>

      </div>
    </main>

    <!-- Footer -->
    <footer class="bg-gray-100 pt-12 pb-8 border-t border-gray-200 mt-auto">
      <div class="container mx-auto px-4 text-center text-gray-500 text-sm">
        <p>&copy; 2025 AliExpress Clone. All rights reserved.</p>
      </div>
    </footer>

    <!-- Cookie Modal -->
    <CookieConsentModal 
      v-if="showCookieModal"
      @accept="handleAcceptCookies"
      @decline="handleDeclineCookies"
    />

  </div>
</template>

<script>
import { ref, onMounted, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import CookieConsentModal from '../components/CookieConsentModal.vue'

export default {
  name: 'HOME',
  components: {
    CookieConsentModal
  },
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    
    const showCookieModal = computed(() => {
      return store.cookie_accepted !== true
    })

    const cartCount = computed(() => {
      return store.cart_items ? store.cart_items.length : 0
    })

    const handleAcceptCookies = () => {
      store.cookie_accepted = true
    }

    const handleDeclineCookies = () => {
      // Optional: Handle decline logic
      store.cookie_accepted = false // Or handle differently
    }

    const handleGoCategories = async () => {
      store.currentPageId = 'CATEGORY_LIST'
      await router.push({ name: 'CATEGORY_LIST' })
    }

    const handleGoDeals = async () => {
      store.currentPageId = 'DEALS_LIST'
      await router.push({ name: 'DEALS_LIST' })
    }

    const handleGoAccount = async () => {
      store.currentPageId = 'ACCOUNT_LOGIN'
      await router.push({ name: 'ACCOUNT_LOGIN' })
    }

    const handleGoCart = async () => {
      store.currentPageId = 'CART_PAGE'
      await router.push({ name: 'CART_PAGE' })
    }

    return {
      showCookieModal,
      cartCount,
      handleAcceptCookies,
      handleDeclineCookies,
      handleGoCategories,
      handleGoDeals,
      handleGoAccount,
      handleGoCart
    }
  }
}
</script>