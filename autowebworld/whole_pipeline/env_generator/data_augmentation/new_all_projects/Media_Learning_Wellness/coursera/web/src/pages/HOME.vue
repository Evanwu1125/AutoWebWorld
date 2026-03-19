<template>
  <div class="min-h-screen flex flex-col bg-gray-50">
    <!-- Cookie Consent Modal -->
    <CookieConsentModal :show="showCookieModal" @accept="handleAcceptCookies" />

    <!-- Navigation -->
    <nav class="bg-white shadow-sm sticky top-0 z-20">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between h-16">
          <div class="flex items-center">
            <!-- Logo -->
            <div id="header-logo-home" class="flex-shrink-0 flex items-center cursor-pointer" @click="goHome">
              <span class="text-2xl font-bold text-blue-700">Coursera</span>
            </div>
            
            <!-- Desktop Menu -->
            <div class="hidden md:ml-8 md:flex md:space-x-8">
              <div 
                id="nav-explore-courses"
                @click="goToDiscovery"
                class="border-transparent text-gray-500 hover:border-blue-500 hover:text-gray-700 inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium cursor-pointer"
              >
                Explore
              </div>

              <!-- Hover Menu -->
              <div 
                id="nav-browse"
                class="relative group inline-flex items-center px-1 pt-1 border-b-2 border-transparent text-sm font-medium text-gray-500 hover:text-gray-700 cursor-pointer"
              >
                <span class="group-hover:text-blue-700">Browse</span>
                <div class="absolute left-0 top-full mt-0 w-48 bg-white rounded-md shadow-lg py-1 hidden group-hover:block z-50">
                  <div 
                    id="nav-browse-courses"
                    @click="goToDiscovery"
                    class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer"
                  >
                    Courses
                  </div>
                </div>
              </div>

              <!-- Learn Dropdown (Click) -->
              <div class="relative">
                <button 
                  id="nav-dropdown-learn"
                  @click="toggleLearnMenu"
                  class="border-transparent text-gray-500 hover:border-blue-500 hover:text-gray-700 inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium h-full"
                >
                  Learn
                </button>
                <div v-if="isLearnMenuOpen" class="absolute left-0 mt-2 w-56 rounded-md shadow-lg bg-white ring-1 ring-black ring-opacity-5 z-50">
                  <div class="py-1" role="menu">
                    <div 
                      id="nav-dropdown-courses"
                      @click="goToDiscovery"
                      class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer"
                    >
                      Courses
                    </div>
                    <div 
                      id="nav-dropdown-specializations"
                      @click="goToSpecializations"
                      class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer"
                    >
                      Specializations
                    </div>
                    <div 
                      id="nav-dropdown-professional-certs"
                      class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer"
                    >
                      Professional Certificates
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Account Menu -->
          <div class="flex items-center">
            <div class="relative ml-3">
              <button 
                id="nav-account"
                @click="toggleAccountMenu"
                class="bg-blue-700 text-white p-2 rounded-full hover:bg-blue-800 focus:outline-none"
              >
                <span class="sr-only">Open user menu</span>
                <svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                </svg>
              </button>
              <div v-if="isAccountMenuOpen" class="absolute right-0 mt-2 w-48 rounded-md shadow-lg bg-white ring-1 ring-black ring-opacity-5 z-50">
                <div class="py-1" role="menu">
                  <div 
                    id="nav-account-log-in"
                    @click="goToLogin"
                    class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer"
                  >
                    Log In
                  </div>
                  <div 
                    id="nav-account-sign-up"
                    class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer"
                  >
                    Sign Up
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </nav>

    <!-- Hero Section -->
    <div class="relative bg-white overflow-hidden">
      <div class="max-w-7xl mx-auto">
        <div class="relative z-10 pb-8 bg-white sm:pb-16 md:pb-20 lg:max-w-2xl lg:w-full lg:pb-28 xl:pb-32">
          <main class="mt-10 mx-auto max-w-7xl px-4 sm:mt-12 sm:px-6 md:mt-16 lg:mt-20 lg:px-8 xl:mt-28">
            <div class="sm:text-center lg:text-left">
              <h1 class="text-4xl tracking-tight font-extrabold text-gray-900 sm:text-5xl md:text-6xl">
                <span class="block xl:inline">Learn without</span>
                <span class="block text-blue-700 xl:inline"> limits</span>
              </h1>
              <p class="mt-3 text-base text-gray-500 sm:mt-5 sm:text-lg sm:max-w-xl sm:mx-auto md:mt-5 md:text-xl lg:mx-0">
                Start, switch, or advance your career with more than 5,000 courses, Professional Certificates, and degrees from world-class universities and companies.
              </p>
              <div class="mt-5 sm:mt-8 sm:flex sm:justify-center lg:justify-start">
                <div class="rounded-md shadow">
                  <div 
                    @click="goToLogin"
                    class="w-full flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-white bg-blue-700 hover:bg-blue-800 md:py-4 md:text-lg md:px-10 cursor-pointer"
                  >
                    Join for Free
                  </div>
                </div>
              </div>
            </div>
          </main>
        </div>
      </div>
      <div class="lg:absolute lg:inset-y-0 lg:right-0 lg:w-1/2">
        <img 
          class="h-56 w-full object-cover sm:h-72 md:h-96 lg:w-full lg:h-full" 
          src="/images/StudentLearning.jpg" 
          alt="Student learning online" 
        />
      </div>
    </div>

    <!-- Feature Section -->
    <div class="py-12 bg-gray-50">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="lg:text-center">
          <h2 class="text-base text-blue-700 font-semibold tracking-wide uppercase">World-class learning</h2>
          <p class="mt-2 text-3xl leading-8 font-extrabold tracking-tight text-gray-900 sm:text-4xl">
            Achieve your goals
          </p>
        </div>
        <div class="mt-10">
          <div class="space-y-10 md:space-y-0 md:grid md:grid-cols-3 md:gap-x-8 md:gap-y-10">
             <!-- Feature 1 -->
            <div class="flex flex-col items-center text-center">
              <div class="flex items-center justify-center h-12 w-12 rounded-md bg-blue-700 text-white text-2xl">
                 🎓
              </div>
              <h3 class="mt-4 text-lg leading-6 font-medium text-gray-900">Learn from the best</h3>
              <p class="mt-2 text-base text-gray-500">
                Courses from top universities like Yale, Michigan, Stanford, and leading companies like Google and IBM.
              </p>
            </div>
             <!-- Feature 2 -->
            <div class="flex flex-col items-center text-center">
              <div class="flex items-center justify-center h-12 w-12 rounded-md bg-blue-700 text-white text-2xl">
                 💻
              </div>
              <h3 class="mt-4 text-lg leading-6 font-medium text-gray-900">Get career-ready</h3>
              <p class="mt-2 text-base text-gray-500">
                Build job-ready skills with Professional Certificates and degrees.
              </p>
            </div>
             <!-- Feature 3 -->
            <div class="flex flex-col items-center text-center">
              <div class="flex items-center justify-center h-12 w-12 rounded-md bg-blue-700 text-white text-2xl">
                 🚀
              </div>
              <h3 class="mt-4 text-lg leading-6 font-medium text-gray-900">Flexible learning</h3>
              <p class="mt-2 text-base text-gray-500">
                Learn at your own pace, 100% online.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
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
    
    const isLearnMenuOpen = ref(false)
    const isAccountMenuOpen = ref(false)

    const showCookieModal = computed(() => !store.cookie_consent_given)

    function handleAcceptCookies() {
      store.cookie_consent_given = true
    }

    function toggleLearnMenu() {
      isLearnMenuOpen.value = !isLearnMenuOpen.value
    }

    function toggleAccountMenu() {
      isAccountMenuOpen.value = !isAccountMenuOpen.value
    }

    async function goHome() {
      store.setCurrentPageId('HOME')
      // Already home, but maybe refresh or do nothing
    }

    async function goToDiscovery() {
      if (!store.cookie_consent_given) return // Guard
      store.setCurrentPageId('COURSE_DISCOVERY')
      await router.push({ name: 'COURSE_DISCOVERY' })
    }

    async function goToSpecializations() {
      if (!store.cookie_consent_given) return // Guard
      store.setCurrentPageId('SPECIALIZATION_LIST')
      await router.push({ name: 'SPECIALIZATION_LIST' })
    }

    async function goToLogin() {
      if (!store.cookie_consent_given) return // Guard
      store.setCurrentPageId('LOGIN')
      await router.push({ name: 'LOGIN' })
    }

    return {
      store,
      showCookieModal,
      isLearnMenuOpen,
      isAccountMenuOpen,
      handleAcceptCookies,
      toggleLearnMenu,
      toggleAccountMenu,
      goHome,
      goToDiscovery,
      goToSpecializations,
      goToLogin
    }
  }
}
</script>