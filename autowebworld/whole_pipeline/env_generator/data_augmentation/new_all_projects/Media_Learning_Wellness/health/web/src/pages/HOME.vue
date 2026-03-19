<template>
  <div class="min-h-screen bg-white flex flex-col">
    <!-- Navigation -->
    <nav class="bg-white border-b border-gray-200 z-20 relative">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between h-16">
          <div class="flex">
            <div class="flex-shrink-0 flex items-center">
              <span class="text-2xl font-bold text-[#005DAA]">Teladoc Health</span>
            </div>
            <div class="hidden sm:ml-6 sm:flex sm:space-x-8">
              <!-- Visits Hover Menu -->
              <div class="relative group h-full flex items-center">
                <button id="nav-visits" class="text-gray-500 hover:text-gray-900 px-3 py-2 rounded-md text-sm font-medium">
                  Visits
                </button>
                <div class="absolute left-0 top-full w-48 bg-white shadow-lg rounded-md py-2 hidden group-hover:block border border-gray-100">
                  <a id="nav-visits-instant" @click="handleInstantVisitHover" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer">
                    Instant Visit
                  </a>
                </div>
              </div>

              <!-- Menu (Clickable) -->
              <div class="relative h-full flex items-center">
                <button id="nav-menu" @click="toggleMenu" class="text-gray-500 hover:text-gray-900 px-3 py-2 rounded-md text-sm font-medium">
                  Services
                </button>
                <div v-if="menuOpen" class="absolute left-0 top-full mt-2 w-48 bg-white shadow-lg rounded-md py-2 border border-gray-100 z-50">
                  <a id="nav-menu-visits" @click="handleMenuVisits" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer">
                    Schedule Visit
                  </a>
                  <a id="nav-menu-benefits" @click="handleMenuBenefits" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer">
                    Benefits
                  </a>
                </div>
              </div>
            </div>
          </div>
          <div class="flex items-center">
             <button class="bg-transparent text-[#005DAA] font-semibold py-2 px-4 border border-[#005DAA] rounded hover:bg-blue-50">
               Sign In
             </button>
          </div>
        </div>
      </div>
    </nav>

    <!-- Hero Section -->
    <div class="relative flex-1 flex items-center justify-center bg-gray-50 overflow-hidden">
      <!-- Background Image using ImageGetter -->
      <div class="absolute inset-0 z-0">
        <img src="/images/Telemedicine.jpg" alt="Doctor consulting patient online" class="w-full h-full object-cover opacity-90" />
        <div class="absolute inset-0 bg-gradient-to-r from-white/90 to-transparent"></div>
      </div>

      <div class="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 w-full">
        <div class="lg:w-1/2">
          <h1 class="text-4xl font-extrabold tracking-tight text-[#005DAA] sm:text-5xl md:text-6xl mb-6">
            Quality healthcare,<br/>anytime, anywhere.
          </h1>
          <p class="mt-4 max-w-2xl text-xl text-gray-600 mb-8">
            Connect with board-certified doctors and licensed therapists from the comfort of your home. 24/7 access to care.
          </p>
          <div class="mt-8 flex gap-4">
            <button
              id="home-cta-login"
              @click="handleLoginDirect"
              class="inline-flex items-center justify-center px-8 py-4 border border-transparent text-lg font-bold rounded-full text-white bg-[#009CDE] hover:bg-[#007bb0] shadow-lg transition-transform transform hover:-translate-y-1 md:py-4 md:text-xl md:px-10"
            >
              Get Started
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'HOME',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const menuOpen = ref(false)

    // Actions
    const handleLoginDirect = async () => {
      // ACT_HOME_GO_TO_LOGIN_DIRECT
      store.setCurrentPageId('LOGIN')
      await router.push({ name: 'LOGIN' })
    }

    const toggleMenu = () => {
      // Open menu to reveal #nav-menu-visits and #nav-menu-benefits
      menuOpen.value = !menuOpen.value
    }

    const handleMenuVisits = async () => {
      // ACT_HOME_GO_TO_VISIT_TYPES_MENU
      store.setCurrentPageId('VISIT_TYPE_SELECTION')
      await router.push({ name: 'VISIT_TYPE_SELECTION' })
    }

    const handleMenuBenefits = async () => {
      // ACT_HOME_GO_TO_BENEFITS_MENU
      store.setCurrentPageId('BENEFITS_OVERVIEW')
      await router.push({ name: 'BENEFITS_OVERVIEW' })
    }

    const handleInstantVisitHover = async () => {
      // ACT_HOME_GO_TO_INSTANT_VISIT_HOVER
      store.setCurrentPageId('INSTANT_VISIT_TRIAGE')
      await router.push({ name: 'INSTANT_VISIT_TRIAGE' })
    }

    return {
      menuOpen,
      handleLoginDirect,
      toggleMenu,
      handleMenuVisits,
      handleMenuBenefits,
      handleInstantVisitHover
    }
  }
}
</script>