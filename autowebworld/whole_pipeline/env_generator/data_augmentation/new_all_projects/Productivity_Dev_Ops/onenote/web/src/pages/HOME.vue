<template>
  <div class="min-h-screen bg-[#F3F2F1] relative overflow-hidden flex flex-col">
    <!-- Cookie Modal -->
    <CookieConsentModal />

    <!-- Navigation Bar -->
    <nav class="bg-white shadow-sm sticky top-0 z-40">
      <div class="max-w-7xl mx-auto px-6 py-4">
        <div class="flex items-center justify-between">
          <!-- Logo -->
          <div class="flex items-center gap-2">
            <div class="w-10 h-10 bg-purple-600 rounded-lg flex items-center justify-center">
              <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
              </svg>
            </div>
            <span class="text-xl font-bold text-gray-900">OneNote</span>
          </div>

          <!-- Navigation Links -->
          <div class="flex items-center gap-6">
            <!-- Notebooks -->
            <button
              id="nav-notebooks-direct"
              @click="handleDirectNav"
              class="text-gray-700 hover:text-purple-600 font-medium transition-colors"
            >
              Notebooks
            </button>

            <!-- Recent Notes with Hover Menu -->
            <div
              class="relative"
              @mouseenter="showRecents = true"
              @mouseleave="showRecents = false"
            >
              <button
                id="nav-recents-menu"
                class="text-gray-700 hover:text-purple-600 font-medium transition-colors"
              >
                Recent Notes
              </button>

              <!-- Hover Dropdown -->
              <div v-if="showRecents" class="absolute top-full left-0 mt-2 bg-white rounded-lg shadow-xl p-2 min-w-[200px] border border-gray-100">
                <button
                  id="nav-recents-menu-item"
                  @click="handleRecentsNav"
                  class="w-full text-left px-4 py-2 hover:bg-purple-50 rounded-md text-gray-700 font-medium"
                >
                  View Recent Notes
                </button>
              </div>
            </div>

            <!-- More Menu -->
            <div class="relative">
              <button
                id="nav-more"
                @click="showQuickMenu = !showQuickMenu"
                class="text-gray-700 hover:text-purple-600 font-medium transition-colors flex items-center gap-1"
              >
                More
                <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
                </svg>
              </button>

              <!-- Dropdown Menu -->
              <div v-if="showQuickMenu" class="absolute top-full right-0 mt-2 bg-white rounded-lg shadow-xl p-2 min-w-[200px] border border-gray-100">
                <button
                  id="nav-quick-notes-item"
                  @click="handleQuickNotesNav"
                  class="w-full text-left px-4 py-2 hover:bg-yellow-50 rounded-md text-gray-700 font-medium flex items-center gap-2"
                >
                  <span>📝</span> Quick Notes
                </button>
              </div>
            </div>

            <!-- Account Menu -->
            <div class="relative">
              <button
                id="nav-account"
                @click="showAccountMenu = !showAccountMenu"
                class="w-10 h-10 bg-gray-200 rounded-full flex items-center justify-center hover:bg-gray-300 transition-colors"
              >
                <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-gray-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                </svg>
              </button>

              <!-- Dropdown Menu -->
              <div v-if="showAccountMenu" class="absolute top-full right-0 mt-2 bg-white rounded-lg shadow-xl p-2 min-w-[200px] border border-gray-100">
                <button
                  id="nav-settings-item"
                  @click="handleSettingsNav"
                  class="w-full text-left px-4 py-2 hover:bg-blue-50 rounded-md text-gray-700 font-medium flex items-center gap-2"
                >
                  <span>⚙️</span> Settings
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </nav>

    <!-- Hero Section -->
    <div class="relative w-full h-[60vh] bg-cover bg-center" style="background-image: url('https://images.unsplash.com/photo-1517842645767-c639042777db?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80')">
      <div class="absolute inset-0 bg-gradient-to-r from-purple-900/90 to-transparent"></div>
      <div class="relative h-full flex flex-col justify-center px-8 md:px-16 max-w-6xl mx-auto">
        <h1 class="text-5xl md:text-7xl font-bold text-white mb-6 drop-shadow-lg">
          Capture <span class="text-purple-300">Every</span> Idea.
        </h1>
        <p class="text-xl md:text-2xl text-gray-200 mb-10 max-w-2xl leading-relaxed">
          Your digital notebook for everything. Organize your thoughts, plans, and discoveries in one beautiful place.
        </p>
        
        <!-- Direct Navigation Action -->
        <button
          id="hero-cta-notebooks"
          @click="handleDirectNav"
          class="w-max bg-white text-purple-900 hover:bg-purple-50 font-bold py-4 px-8 rounded-full shadow-lg transform transition hover:scale-105 flex items-center gap-2 group"
        >
          <span>Get Started</span>
          <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 group-hover:translate-x-1 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3" />
          </svg>
        </button>
      </div>
    </div>

    <!-- Features Section -->
    <!-- <div class="max-w-7xl mx-auto w-full px-6 py-16">
      <div class="text-center mb-12">
        <h2 class="text-3xl md:text-4xl font-bold text-gray-900 mb-4">Everything You Need</h2>
        <p class="text-lg text-gray-600">Powerful features to organize your digital life</p>
      </div> -->

      <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
        <!-- Feature 1 -->
        <!-- <div class="bg-white rounded-xl shadow-sm p-6 hover:shadow-md transition-shadow">
          <div class="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mb-4">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
            </svg>
          </div>
          <h3 class="text-xl font-bold text-gray-900 mb-2">Organized Notebooks</h3>
          <p class="text-gray-600">Keep your notes structured in notebooks and sections</p>
        </div> -->

        <!-- Feature 2 -->
        <!-- <div class="bg-white rounded-xl shadow-sm p-6 hover:shadow-md transition-shadow">
          <div class="w-12 h-12 bg-yellow-100 rounded-lg flex items-center justify-center mb-4">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-yellow-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
          <h3 class="text-xl font-bold text-gray-900 mb-2">Quick Capture</h3>
          <p class="text-gray-600">Jot down ideas instantly without organizing</p>
        </div> -->

        <!-- Feature 3 -->
        <!-- <div class="bg-white rounded-xl shadow-sm p-6 hover:shadow-md transition-shadow">
          <div class="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <h3 class="text-xl font-bold text-gray-900 mb-2">Recent Access</h3>
          <p class="text-gray-600">Jump back into your latest work seamlessly</p>
        </div>
      </div> -->
    </div>
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
    const showRecents = ref(false)
    const showQuickMenu = ref(false)
    const showAccountMenu = ref(false)

    // Action Handlers
    const handleDirectNav = async () => {
      if (store.cookie_consent_given) {
        store.current_page_id = 'NOTEBOOK_LIST'
        await router.push({ name: 'NOTEBOOK_LIST' })
      }
    }

    const handleRecentsNav = async () => {
      if (store.cookie_consent_given) {
        showRecents.value = false
        store.current_page_id = 'RECENT_NOTES'
        await router.push({ name: 'RECENT_NOTES' })
      }
    }

    const handleQuickNotesNav = async () => {
      if (store.cookie_consent_given) {
        showQuickMenu.value = false
        store.current_page_id = 'QUICK_NOTES'
        await router.push({ name: 'QUICK_NOTES' })
      }
    }

    const handleSettingsNav = async () => {
      if (store.cookie_consent_given) {
        showAccountMenu.value = false
        store.current_page_id = 'SETTINGS'
        await router.push({ name: 'SETTINGS' })
      }
    }

    // Cookie accept action is handled within the modal component calling store directly 
    // or via an event. For FSM mapping, the button #cookie-accept is inside the modal.
    // The logic there updates the store directly.

    return {
      showRecents,
      showQuickMenu,
      showAccountMenu,
      handleDirectNav,
      handleRecentsNav,
      handleQuickNotesNav,
      handleSettingsNav
    }
  }
}
</script>