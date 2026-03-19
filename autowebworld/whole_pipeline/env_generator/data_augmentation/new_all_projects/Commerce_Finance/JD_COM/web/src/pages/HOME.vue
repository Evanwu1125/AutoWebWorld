<template>
  <div class="min-h-screen bg-[#F6F6F6] font-sans">
    <!-- Global Components -->
    <CookieConsentModal />

    <!-- Header -->
    <header class="sticky top-0 z-50 bg-white shadow-sm border-b border-gray-200">
      <div class="container mx-auto px-4 h-16 flex items-center justify-between">
        <!-- Logo -->
        <div class="flex items-center gap-2">
          <div class="text-[#E1251B] text-3xl font-black tracking-tighter">JD</div>
          <div class="hidden sm:block text-gray-500 text-xs border-l border-gray-300 pl-2 ml-2 leading-tight">
            Joy<br>Buy
          </div>
        </div>

        <!-- Navigation -->
        <nav class="hidden md:flex items-center space-x-8">
          <button id="nav-electronics" @click="goToElectronics" class="text-gray-700 hover:text-[#E1251B] font-medium transition-colors">
            Electronics
          </button>
          
          <!-- Hover Menu for Supermarket -->
          <div class="relative group" id="nav-categories" @mouseenter="handleHover">
            <button class="text-gray-700 group-hover:text-[#E1251B] font-medium flex items-center gap-1">
              Categories
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
            </button>
            <div class="absolute top-full left-0 w-48 bg-white shadow-xl rounded-lg mt-2 py-2 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 transform origin-top-left z-50">
              <div class="option-electronics px-4 py-2 hover:bg-gray-50 cursor-pointer text-gray-700" @click="goToElectronics">Electronics</div>
              <div class="option-supermarket px-4 py-2 hover:bg-gray-50 cursor-pointer text-gray-700 text-[#E1251B]" @click="goToSupermarket">Supermarket</div>
              <div class="option-fashion px-4 py-2 hover:bg-gray-50 cursor-pointer text-gray-700">Fashion</div>
            </div>
          </div>
        </nav>

        <!-- Right Actions -->
        <div class="flex items-center gap-4">
          <button id="nav-login" @click="goToLogin" class="text-gray-600 hover:text-[#E1251B] text-sm font-medium">
            Sign In / Register
          </button>

          <!-- Menu Toggle -->
          <div class="relative">
            <button id="nav-menu-toggle" @click="toggleMenu" class="p-2 hover:bg-gray-100 rounded-full">
              <svg class="w-6 h-6 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"></path></svg>
            </button>
            
            <!-- Dropdown Menu -->
            <div v-if="menuOpen" id="nav-menu" class="absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-xl py-2 z-50 border border-gray-100 animate-fade-in">
              <div id="nav-menu-home" class="px-4 py-2 hover:bg-gray-50 cursor-pointer text-gray-700">Home</div>
              <div id="nav-menu-cart" @click="goToCart" class="px-4 py-2 hover:bg-gray-50 cursor-pointer text-gray-700 flex justify-between items-center">
                Cart <span class="bg-[#E1251B] text-white text-xs px-1.5 py-0.5 rounded-full">0</span>
              </div>
              <div id="nav-menu-orders" class="px-4 py-2 hover:bg-gray-50 cursor-pointer text-gray-700">My Orders</div>
              <div id="nav-menu-profile" class="px-4 py-2 hover:bg-gray-50 cursor-pointer text-gray-700">Profile</div>
            </div>
          </div>
        </div>
      </div>
    </header>

    <!-- Hero Section -->
    <main>
      <div class="relative h-[400px] sm:h-[500px] w-full overflow-hidden">
         <img src="/images/ShoppingFestival.jpg" 
             alt="Shopping Festival" 
             class="w-full h-full object-cover" />
        <div class="absolute inset-0 bg-gradient-to-r from-black/60 to-transparent flex items-center">
          <div class="container mx-auto px-4">
            <h1 class="text-4xl sm:text-6xl font-bold text-white mb-4 leading-tight">
              Quality Life,<br> Delivered.
            </h1>
            <p class="text-gray-200 text-lg sm:text-xl mb-8 max-w-lg">
              Discover millions of products with same-day delivery and premium service.
            </p>
            <button @click="goToElectronics" class="bg-[#E1251B] hover:bg-[#c91f16] text-white px-8 py-3 rounded-full font-bold text-lg transition-transform transform hover:scale-105 shadow-lg shadow-red-900/50">
              Shop Now
            </button>
          </div>
        </div>
      </div>

      <!-- Features Grid -->
      <div class="container mx-auto px-4 py-12">
        <h2 class="text-2xl font-bold text-gray-800 mb-8">Featured Categories</h2>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-6">
          <div @click="goToElectronics" class="bg-white p-6 rounded-xl shadow-sm hover:shadow-md transition-shadow cursor-pointer group">
            <div class="w-12 h-12 bg-blue-100 text-blue-600 rounded-lg flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
              📱
            </div>
            <h3 class="font-bold text-lg mb-1">Electronics</h3>
            <p class="text-gray-500 text-sm">Phones, Laptops & More</p>
          </div>
          <div @click="goToSupermarket" class="bg-white p-6 rounded-xl shadow-sm hover:shadow-md transition-shadow cursor-pointer group">
            <div class="w-12 h-12 bg-green-100 text-green-600 rounded-lg flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
              🥦
            </div>
            <h3 class="font-bold text-lg mb-1">Supermarket</h3>
            <p class="text-gray-500 text-sm">Fresh Food & Groceries</p>
          </div>
          <div class="bg-white p-6 rounded-xl shadow-sm hover:shadow-md transition-shadow cursor-pointer group">
            <div class="w-12 h-12 bg-purple-100 text-purple-600 rounded-lg flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
              👗
            </div>
            <h3 class="font-bold text-lg mb-1">Fashion</h3>
            <p class="text-gray-500 text-sm">Clothing & Accessories</p>
          </div>
          <div class="bg-white p-6 rounded-xl shadow-sm hover:shadow-md transition-shadow cursor-pointer group">
            <div class="w-12 h-12 bg-orange-100 text-orange-600 rounded-lg flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
              🏠
            </div>
            <h3 class="font-bold text-lg mb-1">Home</h3>
            <p class="text-gray-500 text-sm">Furniture & Decor</p>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { ref } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import CookieConsentModal from '../components/CookieConsentModal.vue';

export default {
  name: 'HOME',
  components: {
    CookieConsentModal
  },
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const menuOpen = ref(false);

    const toggleMenu = () => {
      menuOpen.value = !menuOpen.value;
    };

    const goToElectronics = async () => {
      signatureStore.currentPageId = 'CATEGORY_ELECTRONICS';
      await router.push({ name: 'CATEGORY_ELECTRONICS' });
    };

    const goToSupermarket = async () => {
      signatureStore.currentPageId = 'CATEGORY_SUPERMARKET';
      await router.push({ name: 'CATEGORY_SUPERMARKET' });
    };

    const goToLogin = async () => {
      signatureStore.currentPageId = 'LOGIN';
      await router.push({ name: 'LOGIN' });
    };

    const goToCart = async () => {
      signatureStore.currentPageId = 'CART';
      await router.push({ name: 'CART' });
    };
    
    const handleHover = () => {
      // Logic for hover effect if needed, mostly CSS handled
    };

    return {
      menuOpen,
      toggleMenu,
      goToElectronics,
      goToSupermarket,
      goToLogin,
      goToCart,
      handleHover
    };
  }
}
</script>