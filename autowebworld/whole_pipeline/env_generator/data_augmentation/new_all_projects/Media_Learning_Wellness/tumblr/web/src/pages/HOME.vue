<template>
  <div class="min-h-screen bg-slate-900 text-white font-sans overflow-hidden relative">
    <!-- Cookie Consent Modal (Interceptor) -->
    <CookieConsentModal />

    <!-- Navigation Bar -->
    <nav class="absolute top-0 w-full z-50 px-6 py-6 flex justify-between items-center">
      <!-- Logo -->
      <div class="text-3xl font-bold tracking-tighter cursor-pointer">
        t<span class="text-slate-400">umblr</span>
      </div>

      <!-- Right Side Actions -->
      <div class="flex items-center gap-4">
        <!-- Direct Dashboard Link -->
        <button 
          id="nav-dashboard-direct" 
          @click="goToDashboardDirect"
          class="hidden md:block text-slate-300 hover:text-white font-semibold transition-colors"
        >
          Dashboard
        </button>

        <!-- Hover Menu (Desktop) -->
        <div 
          id="nav-main"
          class="relative group hidden md:block"
          @mouseenter="hoverMenuOpen = true"
          @mouseleave="hoverMenuOpen = false"
        >
          <button class="text-slate-300 hover:text-white font-semibold transition-colors flex items-center gap-1">
            Explore <span class="text-xs">▼</span>
          </button>
          
          <!-- Hover Dropdown -->
          <div v-if="hoverMenuOpen" class="absolute right-0 mt-2 w-48 bg-slate-800 rounded-lg shadow-xl border border-slate-700 overflow-hidden py-1">
            <div id="nav-main-dashboard" @click="goToDashboardHover('dashboard')" class="px-4 py-2 hover:bg-slate-700 cursor-pointer text-sm">Dashboard</div>
            <div id="nav-main-explore" @click="goToDashboardHover('explore')" class="px-4 py-2 hover:bg-slate-700 cursor-pointer text-sm">Explore</div>
            <div id="nav-main-messages" @click="goToDashboardHover('messages')" class="px-4 py-2 hover:bg-slate-700 cursor-pointer text-sm">Messages</div>
          </div>
        </div>

        <!-- Signup Button -->
        <button 
          id="nav-signup" 
          @click="goToSignup"
          class="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-6 rounded-full transition-all transform hover:scale-105"
        >
          Sign up
        </button>

        <!-- Mobile Menu Toggle -->
        <div class="md:hidden relative">
          <button id="nav-menu-toggle" @click="toggleMobileMenu" class="text-white p-2">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16m-7 6h7" />
            </svg>
          </button>

          <!-- Mobile Dropdown -->
          <div v-if="mobileMenuOpen" id="nav-menu" class="absolute right-0 mt-2 w-48 bg-slate-800 rounded-lg shadow-xl border border-slate-700 overflow-hidden py-1 z-50">
            <div id="nav-menu-dashboard" @click="goToDashboardMenu('dashboard')" class="px-4 py-3 hover:bg-slate-700 cursor-pointer border-b border-slate-700">Dashboard</div>
            <div id="nav-menu-explore" @click="goToDashboardMenu('explore')" class="px-4 py-3 hover:bg-slate-700 cursor-pointer border-b border-slate-700">Explore</div>
            <div id="nav-menu-account" @click="goToDashboardMenu('account')" class="px-4 py-3 hover:bg-slate-700 cursor-pointer">Account</div>
          </div>
        </div>
      </div>
    </nav>

    <!-- Hero Section -->
    <main class="relative h-screen flex flex-col justify-center items-center text-center px-4">
      <!-- Background Image -->
      <div class="absolute inset-0 z-0">
        <img src="/images/Background.jpg" alt="Tumblr Vibes" class="w-full h-full object-cover opacity-60" />
        <div class="absolute inset-0 bg-gradient-to-t from-slate-900 via-slate-900/50 to-transparent"></div>
      </div>

      <!-- Hero Content -->
      <div class="relative z-10 max-w-3xl mx-auto space-y-6">
        <h1 class="text-5xl md:text-7xl font-bold tracking-tight mb-4 drop-shadow-lg">
          Come for what you love.
        </h1>
        <p class="text-xl md:text-2xl text-slate-200 mb-8 drop-shadow-md">
          Stay for what you discover.
        </p>
        
        <!-- Search Bar (Decorative / Alternative entry to dashboard) -->
        <div class="w-full max-w-lg mx-auto bg-white rounded-full flex items-center overflow-hidden p-1">
          <div class="pl-4 text-slate-400">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>
          <input 
            type="text" 
            placeholder="Search Tumblr" 
            class="w-full px-4 py-3 outline-none text-slate-800"
            disabled
          />
          <button class="bg-black text-white px-6 py-2 rounded-full font-bold">Go</button>
        </div>
      </div>

      <!-- Footer Links -->
      <div class="absolute bottom-8 flex gap-6 text-sm text-slate-400 font-medium">
        <a href="#" class="hover:text-white">About</a>
        <a href="#" class="hover:text-white">Apps</a>
        <a href="#" class="hover:text-white">Legal</a>
        <a href="#" class="hover:text-white">Privacy</a>
      </div>
    </main>
  </div>
</template>

<script>
import { ref } from 'vue'
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
    const hoverMenuOpen = ref(false)
    const mobileMenuOpen = ref(false)

    // Precondition Checker
    const checkConsent = () => {
      if (store.cookie_consent_given !== true) {
        alert("Please accept cookies first!")
        return false
      }
      return true
    }

    const goToSignup = async () => {
      if (!checkConsent()) return
      store.currentPageId = 'SIGNUP'
      await router.push({ name: 'SIGNUP' })
    }

    const goToDashboardDirect = async () => {
      if (!checkConsent()) return
      store.currentPageId = 'DASHBOARD_FEED'
      await router.push({ name: 'DASHBOARD_FEED' })
    }

    const goToDashboardHover = async (value) => {
      if (!checkConsent()) return
      // Although value isn't used in FSM effects for navigation, we accept it as param
      store.currentPageId = 'DASHBOARD_FEED'
      await router.push({ name: 'DASHBOARD_FEED' })
    }

    const toggleMobileMenu = () => {
      mobileMenuOpen.value = !mobileMenuOpen.value
    }

    const goToDashboardMenu = async (value) => {
      if (!checkConsent()) return
      mobileMenuOpen.value = false
      store.currentPageId = 'DASHBOARD_FEED'
      await router.push({ name: 'DASHBOARD_FEED' })
    }

    return {
      store,
      hoverMenuOpen,
      mobileMenuOpen,
      goToSignup,
      goToDashboardDirect,
      goToDashboardHover,
      goToDashboardMenu,
      toggleMobileMenu
    }
  }
}
</script>