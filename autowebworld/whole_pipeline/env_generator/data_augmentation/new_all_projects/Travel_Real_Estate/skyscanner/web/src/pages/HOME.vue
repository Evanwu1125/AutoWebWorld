<template>
  <div class="min-h-screen bg-slate-50 flex flex-col font-sans">
    <CookieConsentModal />
    
    <!-- Header -->
    <header class="bg-[#002D5C] text-white py-4 px-6 shadow-md relative z-20">
      <div class="max-w-7xl mx-auto flex items-center justify-between">
        <div class="flex items-center gap-8">
          <div id="logo-home" class="text-2xl font-bold tracking-tight cursor-pointer flex items-center gap-2">
            <svg class="w-8 h-8" fill="currentColor" viewBox="0 0 24 24"><path d="M21 16v-2l-8-5V3.5A1.5 1.5 0 0 0 11.5 2 1.5 1.5 0 0 0 10 3.5V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5l8 2.5z"/></svg>
            Skyscanner
          </div>
          <nav class="hidden md:flex gap-6 font-medium">
            <div id="nav-flights-direct" @click="goToFlightsDirect" class="hover:text-blue-300 cursor-pointer transition-colors">Flights</div>
            <div class="group relative">
              <div id="nav-travel" class="hover:text-blue-300 cursor-pointer flex items-center gap-1 transition-colors">
                Travel
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
              </div>
              <!-- Hover Menu -->
              <div class="absolute top-full left-0 w-48 bg-white text-gray-900 rounded-lg shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 transform translate-y-2 group-hover:translate-y-0 z-50 overflow-hidden">
                <div id="nav-flights-hover" @click="goToFlightsHover" class="block px-4 py-3 hover:bg-blue-50 cursor-pointer font-medium border-b border-gray-100">Flights</div>
                <div class="block px-4 py-3 hover:bg-blue-50 cursor-pointer text-gray-400">Hotels (Coming Soon)</div>
              </div>
            </div>
            <!-- More Menu Toggle -->
            <div class="relative">
              <div id="nav-more-toggle" @click="toggleMenu" class="hover:text-blue-300 cursor-pointer flex items-center gap-1 transition-colors">
                More
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
              </div>
              <div v-if="menuOpen" class="absolute top-full right-0 w-56 bg-white text-gray-900 rounded-lg shadow-xl mt-2 z-50 overflow-hidden">
                <div id="nav-more-flights" @click="goToFlightsMenu" class="block px-4 py-3 hover:bg-blue-50 cursor-pointer font-medium border-b border-gray-100">Search Flights</div>
                <div id="nav-more-hotels" @click="goToHotelsMenu" class="block px-4 py-3 hover:bg-blue-50 cursor-pointer font-medium border-b border-gray-100">Search Hotels</div>
                <div id="nav-more-cars" @click="goToCarsMenu" class="block px-4 py-3 hover:bg-blue-50 cursor-pointer font-medium">Car Hire</div>
              </div>
            </div>
          </nav>
        </div>
        <div id="nav-sign-in" @click="goToLogin" class="text-white font-medium hover:text-blue-300 cursor-pointer flex items-center gap-2 px-4 py-2 rounded-full border border-white/20 hover:bg-white/10 transition-all">
          <span>Log In</span>
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 16l-4-4m0 0l4-4m-4 4h14m-5 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h7a3 3 0 013 3v1"/></svg>
        </div>
      </div>
    </header>

    <!-- Hero Section -->
    <main class="flex-grow relative">
      <div class="relative h-[600px] w-full overflow-hidden">
        <img src="/images/Travel.jpg" alt="Beautiful sky travel destination" class="w-full h-full object-cover brightness-75" />
        <div class="absolute inset-0 bg-gradient-to-b from-black/40 to-transparent"></div>
        
        <div class="absolute inset-0 flex flex-col items-center justify-center text-white px-4 text-center">
          <h1 class="text-5xl md:text-7xl font-bold mb-6 tracking-tight drop-shadow-lg">Where to next?</h1>
          <p class="text-xl md:text-2xl mb-12 font-light opacity-90 max-w-2xl">Find exclusive flight deals and explore the world with confidence.</p>
          
          <!-- Search Widget Placeholder/Links -->
          <div class="bg-white/10 backdrop-blur-md p-2 rounded-2xl flex gap-2 max-w-3xl w-full border border-white/20 shadow-2xl">
            <button @click="goToFlightsDirect" class="flex-1 py-4 bg-[#0770E3] hover:bg-[#0660C3] rounded-xl font-bold text-lg transition-all shadow-lg flex items-center justify-center gap-2">
              <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24"><path d="M21 16v-2l-8-5V3.5A1.5 1.5 0 0 0 11.5 2 1.5 1.5 0 0 0 10 3.5V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5l8 2.5z"/></svg>
              Search Flights
            </button>
            <button @click="goToHotelsMenu" class="flex-1 py-4 bg-white/10 hover:bg-white/20 rounded-xl font-semibold text-lg transition-all flex items-center justify-center gap-2">
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4"/></svg>
              Hotels
            </button>
             <button @click="goToCarsMenu" class="flex-1 py-4 bg-white/10 hover:bg-white/20 rounded-xl font-semibold text-lg transition-all flex items-center justify-center gap-2">
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4"/></svg>
              Car Hire
            </button>
          </div>
        </div>
      </div>
      
      <!-- Features Grid -->
      <div class="max-w-7xl mx-auto py-16 px-6 grid grid-cols-1 md:grid-cols-3 gap-8 relative z-10 -mt-20">
        <div class="bg-white p-8 rounded-2xl shadow-xl hover:shadow-2xl transition-all transform hover:-translate-y-1">
          <div class="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 mb-6">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
          </div>
          <h3 class="text-xl font-bold mb-3 text-gray-900">Real-time Deals</h3>
          <p class="text-gray-600 leading-relaxed">We search hundreds of airlines and travel sites to find the best prices instantly.</p>
        </div>
        <div class="bg-white p-8 rounded-2xl shadow-xl hover:shadow-2xl transition-all transform hover:-translate-y-1">
          <div class="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center text-green-600 mb-6">
             <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
          </div>
          <h3 class="text-xl font-bold mb-3 text-gray-900">No Hidden Fees</h3>
          <p class="text-gray-600 leading-relaxed">The price you see is the price you pay. No surprises at checkout.</p>
        </div>
        <div class="bg-white p-8 rounded-2xl shadow-xl hover:shadow-2xl transition-all transform hover:-translate-y-1">
          <div class="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center text-purple-600 mb-6">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
          </div>
          <h3 class="text-xl font-bold mb-3 text-gray-900">Explore Everywhere</h3>
          <p class="text-gray-600 leading-relaxed">Not sure where to go? Use our "Everywhere" search to find inspiration.</p>
        </div>
      </div>
    </main>
    
    <footer class="bg-gray-50 border-t border-gray-200 py-12 px-6">
      <div class="max-w-7xl mx-auto text-center text-gray-400">
        <p>© 2025 Skyscanner Clone. All rights reserved.</p>
      </div>
    </footer>
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
    const menuOpen = ref(false)

    const toggleMenu = () => {
      menuOpen.value = !menuOpen.value
    }

    const goToFlightsDirect = async () => {
      // Logic: if cookie consent given (checked in modal/store), navigate
      if (store.cookie_consent_given) {
        store.currentPageId = 'FLIGHTS_SEARCH'
        await router.push({ name: 'FLIGHTS_SEARCH' })
      }
    }

    const goToFlightsHover = async () => {
       if (store.cookie_consent_given) {
        store.currentPageId = 'FLIGHTS_SEARCH'
        await router.push({ name: 'FLIGHTS_SEARCH' })
      }
    }

    const goToFlightsMenu = async () => {
       if (store.cookie_consent_given) {
        store.currentPageId = 'FLIGHTS_SEARCH'
        await router.push({ name: 'FLIGHTS_SEARCH' })
      }
    }

    const goToHotelsMenu = async () => {
       if (store.cookie_consent_given) {
        store.currentPageId = 'HOTELS_SEARCH'
        await router.push({ name: 'HOTELS_SEARCH' })
      }
    }

    const goToCarsMenu = async () => {
       if (store.cookie_consent_given) {
        store.currentPageId = 'CARS_SEARCH'
        await router.push({ name: 'CARS_SEARCH' })
      }
    }

    const goToLogin = async () => {
       if (store.cookie_consent_given) {
        store.currentPageId = 'LOGIN'
        await router.push({ name: 'LOGIN' })
      }
    }

    return {
      menuOpen,
      toggleMenu,
      goToFlightsDirect,
      goToFlightsHover,
      goToFlightsMenu,
      goToHotelsMenu,
      goToCarsMenu,
      goToLogin
    }
  }
}
</script>