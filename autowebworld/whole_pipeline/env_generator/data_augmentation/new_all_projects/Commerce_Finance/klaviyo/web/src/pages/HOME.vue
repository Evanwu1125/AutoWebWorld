<template>
  <div class="relative min-h-screen bg-white overflow-hidden">
    <!-- Hero Background with ImageGetter -->
    <div class="absolute inset-0 z-0">
      <img 
        src="/images/MarketingOffice.jpg" 
        alt="Marketing Team Office" 
        class="w-full h-full object-cover opacity-90"
      />
      <div class="absolute inset-0 bg-gradient-to-r from-slate-900/90 to-slate-900/40"></div>
    </div>

    <!-- Navigation Bar (Transparent) -->
    <nav class="relative z-20 flex items-center justify-between px-8 py-6 max-w-7xl mx-auto">
      <div class="flex items-center space-x-2">
        <div class="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
          <span class="text-white font-bold text-xl">K</span>
        </div>
        <span class="text-2xl font-bold text-white tracking-tight">Klaviyo</span>
      </div>

      <div class="hidden md:flex items-center space-x-8">
        <!-- Dashboard Link -->
        <button 
          id="nav-dashboard"
          @click="handleNavDashboard"
          class="text-sm font-medium text-slate-200 hover:text-white transition-colors"
        >
          Dashboard
        </button>

        <!-- Campaigns Hover Menu -->
        <div 
          id="nav-campaigns-menu"
          class="relative group"
          @mouseenter="handleCampaignsHover"
        >
          <button class="text-sm font-medium text-slate-200 hover:text-white flex items-center gap-1 py-2">
            Campaigns
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
          </button>
          
          <div class="absolute top-full left-0 mt-1 w-48 bg-white rounded-lg shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 transform origin-top-left p-2">
            <div 
              id="nav-campaigns-menu .option-campaigns"
              @click="handleNavCampaignsOption('campaigns')"
              class="option-campaigns block px-4 py-2 text-sm text-slate-700 hover:bg-slate-50 hover:text-blue-600 rounded-md cursor-pointer"
            >
              All Campaigns
            </div>
            <div 
              class="option-flows block px-4 py-2 text-sm text-slate-700 hover:bg-slate-50 hover:text-blue-600 rounded-md cursor-pointer"
            >
              Flows
            </div>
            <div 
              class="option-forms block px-4 py-2 text-sm text-slate-700 hover:bg-slate-50 hover:text-blue-600 rounded-md cursor-pointer"
            >
              Signup Forms
            </div>
          </div>
        </div>

        <!-- Flows Link -->
        <button 
          id="nav-flows"
          @click="handleNavFlows"
          class="text-sm font-medium text-slate-200 hover:text-white transition-colors"
        >
          Flows
        </button>

        <!-- Lists/Audience Dropdown (Click) -->
        <div class="relative">
          <button 
            id="nav-audience-dropdown"
            @click="toggleAudienceDropdown"
            class="text-sm font-medium text-slate-200 hover:text-white flex items-center gap-1"
          >
            Audience
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
          </button>
          
          <div v-if="audienceDropdownOpen" class="absolute top-full left-0 mt-2 w-56 bg-white rounded-lg shadow-xl p-2 z-50 animate-in fade-in slide-in-from-top-2">
            <div 
              id="nav-audience-lists-segments"
              @click="handleNavListsOption('lists_segments')"
              class="block px-4 py-2 text-sm text-slate-700 hover:bg-slate-50 hover:text-blue-600 rounded-md cursor-pointer"
            >
              Lists & Segments
            </div>
            <div 
              id="nav-audience-profiles"
              class="block px-4 py-2 text-sm text-slate-700 hover:bg-slate-50 hover:text-blue-600 rounded-md cursor-pointer"
            >
              Profiles
            </div>
            <div 
              id="nav-audience-preferences"
              class="block px-4 py-2 text-sm text-slate-700 hover:bg-slate-50 hover:text-blue-600 rounded-md cursor-pointer"
            >
              Preferences
            </div>
          </div>
        </div>

        <!-- Forms Hover Menu -->
        <div 
          id="nav-forms-menu"
          class="relative group"
          @mouseenter="handleFormsHover"
        >
          <button class="text-sm font-medium text-slate-200 hover:text-white flex items-center gap-1 py-2">
            Content
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
          </button>
          
          <div class="absolute top-full left-0 mt-1 w-48 bg-white rounded-lg shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 transform origin-top-left p-2">
            <div 
              id="nav-forms-menu .option-signup-forms"
              @click="handleNavFormsOption('signup_forms')"
              class="option-signup-forms block px-4 py-2 text-sm text-slate-700 hover:bg-slate-50 hover:text-blue-600 rounded-md cursor-pointer"
            >
              Signup Forms
            </div>
            <div 
              class="option-preferences block px-4 py-2 text-sm text-slate-700 hover:bg-slate-50 hover:text-blue-600 rounded-md cursor-pointer"
            >
              Preference Pages
            </div>
          </div>
        </div>
      </div>

      <!-- Auth Buttons (Decorative) -->
      <div class="flex items-center space-x-4">
        <button class="text-sm font-medium text-white hover:text-blue-200">Log In</button>
        <button class="bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold px-4 py-2 rounded-lg transition-colors">Sign Up</button>
      </div>
    </nav>

    <!-- Hero Content -->
    <main class="relative z-10 flex flex-col items-center justify-center min-h-[80vh] px-4 text-center">
      <h1 class="text-5xl md:text-7xl font-bold text-white mb-6 leading-tight tracking-tight drop-shadow-lg">
        Turn Data Into <br/>
        <span class="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">Growth</span>
      </h1>
      <p class="text-xl md:text-2xl text-slate-200 mb-10 max-w-2xl leading-relaxed drop-shadow-md">
        The unified customer platform for email, SMS, and more. 
        Drive revenue with data-driven marketing automation.
      </p>
      
      <div class="flex flex-col sm:flex-row gap-4">
        <button 
          @click="handleNavDashboard"
          class="bg-white text-blue-900 font-bold py-4 px-8 rounded-full shadow-lg hover:shadow-xl hover:scale-105 transition-all duration-200 text-lg"
        >
          Get Started
        </button>
        <button class="bg-transparent border-2 border-white text-white font-bold py-4 px-8 rounded-full hover:bg-white/10 transition-all duration-200 text-lg">
          Request Demo
        </button>
      </div>

      <!-- Social Proof -->
      <div class="mt-16 pt-8 border-t border-white/20 w-full max-w-4xl">
        <p class="text-sm text-slate-400 uppercase tracking-widest mb-6 font-semibold">Trusted by leading brands</p>
        <div class="flex flex-wrap justify-center gap-8 md:gap-16 opacity-70 grayscale hover:grayscale-0 transition-all duration-500">
          <img src="/images/BrandLogos.jpg" alt="Brand 1" class="h-8 w-auto" />
          <img src="/images/Brand.jpg" alt="Brand 2" class="h-8 w-auto" />
          <img src="/images/Brand.jpg" alt="Brand 3" class="h-8 w-auto" />
          <img src="/images/Brand.jpg" alt="Brand 4" class="h-8 w-auto" />
        </div>
      </div>
    </main>

    <!-- Cookie Consent Modal -->
    <CookieConsentModal 
      :is-visible="showCookieModal" 
      @accept="handleAcceptCookies"
    />
  </div>
</template>

<script>
import { ref, computed } from 'vue'
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
    const audienceDropdownOpen = ref(false)

    const showCookieModal = computed(() => store.cookie_consent_given === null)

    function handleAcceptCookies() {
      store.cookie_consent_given = true
    }

    async function handleNavDashboard() {
      if (!store.cookie_consent_given) return
      store.setCurrentPageId('DASHBOARD')
      await router.push({ name: 'DASHBOARD' })
    }

    function handleCampaignsHover() {
      // Just for UI effect, no state change needed for hover in FSM unless specified
    }

    async function handleNavCampaignsOption(value) {
      if (!store.cookie_consent_given) return
      if (value === 'campaigns') {
        store.setCurrentPageId('CAMPAIGNS_LIST')
        await router.push({ name: 'CAMPAIGNS_LIST' })
      }
      // Other options might route elsewhere or same
    }

    function toggleAudienceDropdown() {
      audienceDropdownOpen.value = !audienceDropdownOpen.value
    }

    async function handleNavListsOption(value) {
      if (!store.cookie_consent_given) return
      if (value === 'lists_segments') {
        store.setCurrentPageId('LISTS_SEGMENTS')
        await router.push({ name: 'LISTS_SEGMENTS' })
      }
    }

    async function handleNavFlows() {
      if (!store.cookie_consent_given) return
      store.setCurrentPageId('FLOWS_LIST')
      await router.push({ name: 'FLOWS_LIST' })
    }
    
    function handleFormsHover() {}

    async function handleNavFormsOption(value) {
      if (!store.cookie_consent_given) return
      if (value === 'signup_forms') {
        store.setCurrentPageId('SIGNUP_FORMS_LIST')
        await router.push({ name: 'SIGNUP_FORMS_LIST' })
      }
    }

    return {
      showCookieModal,
      handleAcceptCookies,
      handleNavDashboard,
      handleCampaignsHover,
      handleNavCampaignsOption,
      audienceDropdownOpen,
      toggleAudienceDropdown,
      handleNavListsOption,
      handleNavFlows,
      handleFormsHover,
      handleNavFormsOption
    }
  }
}
</script>