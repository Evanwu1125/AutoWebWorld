<template>
  <div class="min-h-screen flex flex-col bg-gray-50 text-gray-900 font-sans">
    <!-- Cookie Consent Modal (Interceptor) -->
    <div v-if="showCookieConsent" class="fixed inset-0 z-[10000] flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div class="bg-white rounded-xl shadow-2xl p-8 max-w-md w-full mx-4 animate-fade-in">
        <div class="text-center mb-6">
          <div class="text-4xl mb-4">🍪</div>
          <h2 class="text-2xl font-bold mb-2">We Value Your Privacy</h2>
          <p class="text-gray-600 text-sm leading-relaxed">
            We use cookies to enhance your browsing experience, serve personalized content, and analyze our traffic. By clicking 'Accept All', you consent to our use of cookies.
          </p>
        </div>
        <button 
          id="cookie-accept" 
          @click="acceptCookies" 
          class="w-full bg-[#008060] hover:bg-[#004C3F] text-white font-semibold py-3 px-6 rounded-lg transition-colors duration-200"
        >
          Accept All
        </button>
      </div>
    </div>

    <!-- Navigation -->
    <nav class="bg-white border-b border-gray-200 sticky top-0 z-50">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between h-16">
          <div class="flex items-center">
            <span class="text-xl font-bold text-[#008060] tracking-tight">STOREFRONT</span>
          </div>
          
          <!-- Desktop Nav Dropdown -->
          <div class="hidden md:flex items-center space-x-6">
            <!-- Click Dropdown Menu -->
            <div class="relative">
              <button
                id="nav-dropdown"
                @click="toggleDesktopMenu"
                class="flex items-center gap-2 text-gray-900 font-medium cursor-pointer px-4 py-2 rounded-md hover:bg-gray-100 transition-colors"
              >
                Menu
                <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
                </svg>
              </button>

              <!-- Desktop Dropdown Container -->
              <div
                v-if="desktopMenuOpen"
                id="nav-dropdown"
                class="absolute top-full mt-2 left-0 bg-white border border-gray-200 rounded-lg shadow-lg py-2 min-w-[200px] z-50"
              >
                <div
                  id="nav-dropdown-home"
                  class="block px-4 py-2 text-gray-900 hover:bg-gray-100 cursor-pointer"
                >
                  Home
                </div>
                <div
                  id="nav-dropdown-collections"
                  @click="handleDesktopMenuNav('collections')"
                  class="block px-4 py-2 text-gray-500 hover:bg-gray-100 hover:text-[#008060] cursor-pointer transition-colors"
                >
                  Collections
                </div>
                <div
                  id="nav-dropdown-contact"
                  class="block px-4 py-2 text-gray-500 hover:bg-gray-100 cursor-pointer"
                >
                  Contact
                </div>
              </div>
            </div>

            <!-- Hover Menu -->
            <div
              class="relative"
              @mouseenter="showHoverMenu"
              @mouseleave="hideHoverMenu"
            >
              <div
                id="main-menu"
                class="flex items-center gap-2 text-gray-900 font-medium cursor-pointer px-4 py-2 rounded-md hover:bg-gray-100 transition-colors"
              >
                Navigation
                <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
                </svg>
              </div>

              <!-- Hover Menu Container -->
              <div
                v-if="hoverMenuOpen"
                id="main-menu"
                class="absolute top-full mt-2 left-0 bg-white border border-gray-200 rounded-lg shadow-lg py-2 min-w-[200px] z-50"
              >
                <div
                  id="menu-link-home"
                  class="block px-4 py-2 text-gray-900 hover:bg-gray-100 cursor-pointer"
                >
                  Home
                </div>
                <div
                  id="menu-link-collections"
                  @click="handleHoverNav('collections')"
                  class="block px-4 py-2 text-gray-500 hover:bg-gray-100 hover:text-[#008060] cursor-pointer transition-colors"
                >
                  Collections
                </div>
                <div
                  id="menu-link-about"
                  class="block px-4 py-2 text-gray-500 hover:bg-gray-100 cursor-pointer"
                >
                  About
                </div>
              </div>
            </div>
          </div>

          <div class="flex items-center space-x-4">
             <button 
              id="nav-login" 
              @click="goToLogin"
              class="text-gray-500 hover:text-[#008060] font-medium transition-colors"
            >
              Log In
            </button>
             <button 
              id="nav-dropdown"
              class="md:hidden p-2"
              @click="toggleMobileMenu"
             >
                <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16" />
                </svg>
             </button>
          </div>
        </div>
      </div>
      
      <!-- Mobile Dropdown -->
      <div v-if="mobileMenuOpen" id="nav-dropdown" class="md:hidden bg-white border-t border-gray-200 p-4 space-y-2 shadow-lg absolute w-full z-40">
         <div id="nav-dropdown-home" class="block p-2 text-gray-900">Home</div>
         <div 
           id="nav-dropdown-collections" 
           @click="handleMenuNav('collections')"
           class="block p-2 text-gray-500 hover:text-[#008060] cursor-pointer"
         >
           Collections
         </div>
         <div id="nav-dropdown-contact" class="block p-2 text-gray-500">Contact</div>
      </div>
    </nav>

    <!-- Hero Section -->
    <div class="relative bg-gray-900 h-[600px] flex items-center overflow-hidden">
      <!-- ImageGetter: Hero Background -->
      <div class="absolute inset-0 z-0 opacity-60">
        <img 
            :src="'/images/FashionStore.jpg'" 
            alt="Modern store interior" 
            class="w-full h-full object-cover"
        />
      </div>
      
      <div class="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <h1 class="text-5xl md:text-7xl font-extrabold text-white mb-6 tracking-tight drop-shadow-lg">
          Elevate Your Lifestyle
        </h1>
        <p class="text-xl text-gray-200 mb-10 max-w-2xl mx-auto drop-shadow-md">
          Discover our curated collection of premium products designed for modern living.
        </p>
        <button 
          id="nav-shop-all" 
          @click="goToCollections"
          class="bg-[#008060] hover:bg-[#004C3F] text-white font-bold py-4 px-10 rounded-full text-lg shadow-lg hover:shadow-xl transform hover:-translate-y-1 transition-all duration-300"
        >
          Shop All Collections
        </button>
      </div>
    </div>

    <!-- Featured Section (Decorative) -->
    <div class="py-24 bg-white">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <h2 class="text-3xl font-bold text-gray-900 mb-12 text-center">Featured Categories</h2>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
             <div class="group cursor-pointer">
                <div class="relative h-96 rounded-2xl overflow-hidden mb-4 shadow-md">
                    <img :src="'/images/Electronics.jpg'" alt="Electronics" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" />
                    <div class="absolute inset-0 bg-black/20 group-hover:bg-black/10 transition-colors"></div>
                    <div class="absolute bottom-6 left-6 text-white text-2xl font-bold">Electronics</div>
                </div>
             </div>
             <div class="group cursor-pointer">
                <div class="relative h-96 rounded-2xl overflow-hidden mb-4 shadow-md">
                    <img :src="'/images/Furniture.jpg'" alt="Furniture" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" />
                    <div class="absolute inset-0 bg-black/20 group-hover:bg-black/10 transition-colors"></div>
                    <div class="absolute bottom-6 left-6 text-white text-2xl font-bold">Furniture</div>
                </div>
             </div>
             <div class="group cursor-pointer">
                <div class="relative h-96 rounded-2xl overflow-hidden mb-4 shadow-md">
                    <img :src="'/images/Clothing.jpg'" alt="Clothing" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" />
                    <div class="absolute inset-0 bg-black/20 group-hover:bg-black/10 transition-colors"></div>
                    <div class="absolute bottom-6 left-6 text-white text-2xl font-bold">Clothing</div>
                </div>
             </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { computed, ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'HOME',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const mobileMenuOpen = ref(false)
    const desktopMenuOpen = ref(false)
    const hoverMenuOpen = ref(false)

    const showCookieConsent = computed(() => signatureStore.cookie_consent_given === null)

    const acceptCookies = () => {
      signatureStore.cookie_consent_given = true
    }

    const goToCollections = async () => {
      if (!signatureStore.cookie_consent_given) return
      signatureStore.currentPageId = 'SHOP_ALL_COLLECTIONS'
      await router.push({ name: 'SHOP_ALL_COLLECTIONS' })
    }

    const handleHoverNav = async (target) => {
       if (!signatureStore.cookie_consent_given) return
       if (target === 'collections') {
          signatureStore.currentPageId = 'SHOP_ALL_COLLECTIONS'
          hoverMenuOpen.value = false
          await router.push({ name: 'SHOP_ALL_COLLECTIONS' })
       }
    }

    const handleMenuNav = async (target) => {
       if (!signatureStore.cookie_consent_given) return
       if (target === 'collections') {
          signatureStore.currentPageId = 'SHOP_ALL_COLLECTIONS'
          await router.push({ name: 'SHOP_ALL_COLLECTIONS' })
       }
    }

    const handleDesktopMenuNav = async (target) => {
       if (!signatureStore.cookie_consent_given) return
       if (target === 'collections') {
          signatureStore.currentPageId = 'SHOP_ALL_COLLECTIONS'
          desktopMenuOpen.value = false
          await router.push({ name: 'SHOP_ALL_COLLECTIONS' })
       }
    }

    const goToLogin = async () => {
      if (!signatureStore.cookie_consent_given) return
      signatureStore.currentPageId = 'CUSTOMER_LOGIN'
      await router.push({ name: 'CUSTOMER_LOGIN' })
    }

    const toggleMobileMenu = () => {
        mobileMenuOpen.value = !mobileMenuOpen.value
    }

    const toggleDesktopMenu = () => {
        desktopMenuOpen.value = !desktopMenuOpen.value
    }

    const showHoverMenu = () => {
        hoverMenuOpen.value = true
    }

    const hideHoverMenu = () => {
        hoverMenuOpen.value = false
    }

    return {
      showCookieConsent,
      acceptCookies,
      goToCollections,
      handleHoverNav,
      handleMenuNav,
      handleDesktopMenuNav,
      goToLogin,
      toggleMobileMenu,
      toggleDesktopMenu,
      showHoverMenu,
      hideHoverMenu,
      mobileMenuOpen,
      desktopMenuOpen,
      hoverMenuOpen
    }
  }
}
</script>