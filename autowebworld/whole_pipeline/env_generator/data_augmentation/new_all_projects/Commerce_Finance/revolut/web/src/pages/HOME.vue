<template>
  <div class="min-h-screen bg-gray-100 flex flex-col items-center pt-10 pb-20 px-4">
    <!-- Hero Section -->
    <div class="w-full max-w-md bg-white rounded-3xl shadow-xl overflow-hidden mb-6 relative">
      <!-- Background Image -->
      <div class="absolute inset-0 z-0">
        <img :src="heroImage" class="w-full h-full object-cover opacity-10" alt="Revolut Hero" />
      </div>
      
      <!-- Content -->
      <div class="relative z-10 p-8 flex flex-col items-center text-center">
        <div class="w-16 h-16 bg-blue-600 rounded-2xl flex items-center justify-center mb-4 shadow-lg shadow-blue-200">
          <span class="text-3xl font-bold text-white">R</span>
        </div>
        <h1 class="text-3xl font-extrabold text-gray-900 mb-2">Welcome Back</h1>
        <p class="text-gray-500 mb-8">Manage your money with ease and security.</p>
        
        <div class="grid grid-cols-2 gap-4 w-full">
          <!-- Accounts Direct -->
          <button 
            id="tab-accounts"
            @click="goToAccounts"
            class="flex flex-col items-center justify-center p-4 bg-gray-50 hover:bg-blue-50 rounded-2xl transition-all group border border-transparent hover:border-blue-100"
          >
            <div class="w-12 h-12 bg-white rounded-full shadow-sm flex items-center justify-center mb-2 group-hover:scale-110 transition-transform">
              <svg class="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z"></path></svg>
            </div>
            <span class="font-semibold text-gray-700 group-hover:text-blue-700">Accounts</span>
          </button>

          <!-- Exchange Direct -->
          <button 
            id="quick-action-exchange"
            @click="goToExchange"
            class="flex flex-col items-center justify-center p-4 bg-gray-50 hover:bg-green-50 rounded-2xl transition-all group border border-transparent hover:border-green-100"
          >
            <div class="w-12 h-12 bg-white rounded-full shadow-sm flex items-center justify-center mb-2 group-hover:scale-110 transition-transform">
              <svg class="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4"></path></svg>
            </div>
            <span class="font-semibold text-gray-700 group-hover:text-green-700">Exchange</span>
          </button>
        </div>
      </div>
    </div>

    <!-- Cards Hover Menu -->
    <div 
      id="header-cards"
      class="w-full max-w-md bg-white rounded-2xl shadow-md p-4 mb-4 relative group cursor-pointer hover:shadow-lg transition-shadow"
      @mouseenter="showCardsMenu = true"
      @mouseleave="showCardsMenu = false"
    >
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-3">
          <div class="bg-pink-100 p-2 rounded-lg">
            <svg class="w-6 h-6 text-pink-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z"></path></svg>
          </div>
          <span class="font-bold text-lg">My Cards</span>
        </div>
        <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
      </div>

      <!-- Dropdown -->
      <div v-if="showCardsMenu" class="absolute top-full left-0 w-full mt-2 bg-white rounded-xl shadow-xl z-50 overflow-hidden border border-gray-100">
        <div 
          id="header-cards-physical" 
          @click="selectCardsType('physical_cards')"
          class="p-4 hover:bg-gray-50 flex items-center gap-3 border-b border-gray-50"
        >
          <span class="w-2 h-2 rounded-full bg-blue-500"></span>
          <span>Physical Cards</span>
        </div>
        <div 
          id="header-cards-virtual" 
          @click="selectCardsType('virtual_cards')"
          class="p-4 hover:bg-gray-50 flex items-center gap-3"
        >
          <span class="w-2 h-2 rounded-full bg-pink-500"></span>
          <span>Virtual Cards</span>
        </div>
      </div>
    </div>

    <!-- Transfers Menu -->
    <div class="w-full max-w-md">
      <button 
        id="menu-more"
        @click="toggleMoreMenu"
        class="w-full bg-white rounded-2xl shadow-md p-4 flex items-center justify-between hover:bg-gray-50 transition-colors"
      >
        <div class="flex items-center gap-3">
          <div class="bg-purple-100 p-2 rounded-lg">
            <svg class="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"></path></svg>
          </div>
          <span class="font-bold text-lg">More Actions</span>
        </div>
        <svg class="w-5 h-5 text-gray-400 transform transition-transform" :class="{ 'rotate-180': showMoreMenu }" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
      </button>

      <div v-if="showMoreMenu" class="mt-2 bg-white rounded-2xl shadow-lg p-2 animate-fade-in-down">
        <button 
          id="menu-item-transfers"
          @click="goToTransfers"
          class="w-full p-3 text-left hover:bg-gray-50 rounded-xl flex items-center gap-3"
        >
          <div class="bg-blue-100 p-2 rounded-full">
            <svg class="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 9V7a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2m2 4h10a2 2 0 002-2v-6a2 2 0 00-2-2H9a2 2 0 00-2 2v6a2 2 0 002 2zm7-5a2 2 0 11-4 0 2 2 0 014 0z"></path></svg>
          </div>
          <span class="font-medium">Transfers & Payments</span>
        </button>
      </div>
    </div>

    <!-- Cookie Consent Modal -->
    <div v-if="showCookieConsent" class="fixed inset-0 z-[10000] flex items-end sm:items-center justify-center bg-black/40 backdrop-blur-sm p-4">
      <div class="bg-white rounded-2xl shadow-2xl p-6 w-full max-w-md transform transition-all animate-slide-up">
        <div class="flex items-start gap-4 mb-4">
          <div class="text-4xl">🍪</div>
          <div>
            <h3 class="text-xl font-bold text-gray-900">We Value Your Privacy</h3>
            <p class="text-gray-500 text-sm mt-1">We use cookies to ensure you get the best experience on our website.</p>
          </div>
        </div>
        <button 
          id="cookie-accept"
          @click="acceptCookies"
          class="w-full py-3 bg-gray-900 hover:bg-black text-white font-bold rounded-xl transition-colors"
        >
          Accept All Cookies
        </button>
      </div>
    </div>

  </div>
</template>

<script>
import { ref, onMounted, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
// ImageGetter tool will replace this placeholder
import heroImg from '/images/HeroImage.jpg' 

export default {
  name: 'HOME',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const showCardsMenu = ref(false)
    const showMoreMenu = ref(false)
    
    // We'll use a computed property or ref for the image to allow tool replacement
    const heroImage = ref('/images/Fintech.jpg') 

    const showCookieConsent = computed(() => {
      return signatureStore.cookie_consent_given !== true
    })

    const acceptCookies = () => {
      signatureStore.cookie_consent_given = true
    }

    const goToAccounts = () => {
      signatureStore.setCurrentPageId('ACCOUNTS_DASHBOARD')
      router.push({ name: 'ACCOUNTS_DASHBOARD' })
    }

    const goToExchange = () => {
      signatureStore.setCurrentPageId('EXCHANGE_DASHBOARD')
      router.push({ name: 'EXCHANGE_DASHBOARD' })
    }

    const toggleMoreMenu = () => {
      showMoreMenu.value = !showMoreMenu.value
    }

    const goToTransfers = () => {
      signatureStore.setCurrentPageId('PAYMENTS_LIST')
      router.push({ name: 'PAYMENTS_LIST' })
    }

    const selectCardsType = (type) => {
      // In a real app we might filter based on type, but FSM just says go to CARDS_LIST
      // We could store the selection if needed, but FSM doesn't specify a field for it in HOME->CARDS transition params explicitly for state, 
      // but it does have widget: "hover_menu". The FSM effects are empty.
      // So we just navigate.
      signatureStore.setCurrentPageId('CARDS_LIST')
      router.push({ name: 'CARDS_LIST' })
    }

    return {
      heroImage,
      showCardsMenu,
      showMoreMenu,
      showCookieConsent,
      acceptCookies,
      goToAccounts,
      goToExchange,
      toggleMoreMenu,
      goToTransfers,
      selectCardsType
    }
  }
}
</script>

<style scoped>
.animate-fade-in-down {
  animation: fadeInDown 0.3s ease-out;
}
.animate-slide-up {
  animation: slideUp 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}

@keyframes fadeInDown {
  from { opacity: 0; transform: translateY(-10px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes slideUp {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>