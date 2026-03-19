<template>
  <div class="min-h-screen flex flex-col">
    <!-- Navigation -->
    <nav class="bg-white border-b border-gray-200 px-6 py-4 flex justify-between items-center relative z-20">
      <div class="flex items-center space-x-8">
        <div class="text-blue-600 text-2xl font-bold">Zoom Clone</div>
        
        <!-- Hover Menu Action -->
        <div class="relative group" id="top-nav-products">
          <button class="text-gray-600 hover:text-blue-600 font-medium">Products</button>
          <div class="absolute hidden group-hover:block bg-white shadow-lg border border-gray-100 rounded-lg p-2 w-48 mt-2">
            <button 
              id="top-nav-products-zoom-client"
              @click="handleHoverNav"
              class="block w-full text-left px-4 py-2 hover:bg-blue-50 text-gray-700 hover:text-blue-600 rounded-md"
            >
              Zoom Client
            </button>
          </div>
        </div>
        
        <a href="#" class="text-gray-600 hover:text-blue-600 font-medium">Solutions</a>
        <a href="#" class="text-gray-600 hover:text-blue-600 font-medium">Resources</a>
        <a href="#" class="text-gray-600 hover:text-blue-600 font-medium">Plans & Pricing</a>
      </div>

      <div class="flex items-center space-x-4">
        <!-- Menu Action -->
        <div class="relative">
          <button 
            id="home-profile-menu-toggle"
            @click="toggleMenu"
            class="text-gray-600 hover:text-blue-600 font-medium flex items-center"
          >
            My Account <span class="ml-1">▼</span>
          </button>
          <div v-if="menuOpen" class="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg py-1 z-50 ring-1 ring-black ring-opacity-5">
            <button
              id="home-profile-menu-my-account"
              @click="handleMenuNav"
              class="block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
            >
              Dashboard
            </button>
          </div>
        </div>
      </div>
    </nav>

    <!-- Hero Section -->
    <main class="flex-grow relative flex items-center justify-center bg-gray-50 overflow-hidden">
      <div class="absolute inset-0 z-0">
         <!-- Use ImageGetter for background -->
         <img src="/images/OfficeCollaboration.jpg" alt="Office collaboration background" class="w-full h-full object-cover opacity-20" />
      </div>
      
      <div class="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row items-center">
        <div class="md:w-1/2 text-center md:text-left mb-10 md:mb-0">
          <h1 class="text-4xl md:text-6xl font-bold text-gray-900 leading-tight mb-6">
            One platform to <br/><span class="text-blue-600">connect</span>
          </h1>
          <p class="text-xl text-gray-600 mb-8 max-w-lg mx-auto md:mx-0">
            Bring teams together, reimagine workspaces, engage new audiences, and delight your customers — all on the Zoom platform you know and love.
          </p>
          
          <!-- Direct Action -->
          <div class="flex flex-col sm:flex-row gap-4 justify-center md:justify-start">
            <button 
              id="home-start-zooming-button"
              @click="handleDirectNav"
              class="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-8 rounded-full text-lg transition-all transform hover:scale-105 shadow-lg"
            >
              Start Zooming
            </button>
            <button class="bg-transparent border-2 border-gray-300 hover:border-gray-400 text-gray-700 font-semibold py-3 px-8 rounded-full text-lg transition-all">
              Contact Sales
            </button>
          </div>
        </div>
        
        <div class="md:w-1/2 relative">
          <img src="/images/VideoConferencing.jpg" alt="Video conferencing illustration" class="w-full h-auto drop-shadow-2xl rounded-lg transform rotate-2 hover:rotate-0 transition-all duration-500" />
        </div>
      </div>
    </main>
    
    <!-- Footer stub -->
    <footer class="bg-gray-900 text-white py-12">
      <div class="max-w-7xl mx-auto px-4 text-center text-gray-400">
        &copy; 2025 Zoom Video Communications, Inc. All rights reserved.
      </div>
    </footer>
  </div>
</template>

<script>
import { ref } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'HOME',
  setup() {
    const router = useRouter();
    const store = useSignatureStore();
    const menuOpen = ref(false);

    const toggleMenu = () => {
      menuOpen.value = !menuOpen.value;
    };

    const handleDirectNav = async () => {
      if (store.handleAction('ACT_HOME_GO_TO_DASHBOARD_DIRECT')) {
        await router.push({ name: 'DASHBOARD' });
      }
    };

    const handleHoverNav = async () => {
      if (store.handleAction('ACT_HOME_GO_TO_DASHBOARD_HOVER')) {
        await router.push({ name: 'DASHBOARD' });
      }
    };

    const handleMenuNav = async () => {
      if (store.handleAction('ACT_HOME_GO_TO_DASHBOARD_MENU')) {
        await router.push({ name: 'DASHBOARD' });
      }
    };

    return {
      menuOpen,
      toggleMenu,
      handleDirectNav,
      handleHoverNav,
      handleMenuNav
    };
  }
}
</script>