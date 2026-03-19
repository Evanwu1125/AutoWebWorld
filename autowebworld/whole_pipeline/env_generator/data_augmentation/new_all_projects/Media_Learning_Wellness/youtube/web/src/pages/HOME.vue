<template>
  <div class="min-h-screen bg-[#0F0F0F] text-white flex flex-col relative overflow-hidden">
    <!-- Cookie Consent Modal -->
    <CookieConsentModal />

    <!-- Navbar -->
    <nav class="sticky top-0 z-50 bg-[#0F0F0F]/95 backdrop-blur border-b border-gray-800 px-4 h-14 flex items-center justify-between">
      <!-- Left: Logo and Menu -->
      <div class="flex items-center gap-4">
        <button class="p-2 hover:bg-gray-800 rounded-full">
          <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"></path></svg>
        </button>
        <div id="logo-home" @click="refreshHome" class="flex items-center gap-1 cursor-pointer">
          <div class="bg-red-600 text-white rounded-lg p-1">
            <svg class="w-6 h-6 fill-current" viewBox="0 0 24 24"><path d="M19.615 3.184c-3.604-.246-11.631-.245-15.23 0-3.897.266-4.356 2.62-4.385 8.816.029 6.185.484 8.549 4.385 8.816 3.6.245 11.626.246 15.23 0 3.897-.266 4.356-2.62 4.385-8.816-.029-6.185-.484-8.549-4.385-8.816zm-10.615 12.816v-8l8 3.993-8 4.007z"/></svg>
          </div>
          <span class="text-xl font-bold tracking-tight">YouTube</span>
        </div>
      </div>

      <!-- Center: Search (Nav only) -->
      <div class="flex-1 max-w-2xl mx-4 hidden md:flex">
        <div 
          id="nav-search-results" 
          @click="goSearchResults"
          class="flex w-full cursor-pointer group"
        >
          <div class="flex-1 bg-[#121212] border border-gray-700 rounded-l-full px-4 py-2 text-gray-400 group-hover:border-blue-500 transition-colors flex items-center">
            Search
          </div>
          <button class="bg-[#222] border border-l-0 border-gray-700 rounded-r-full px-6 hover:bg-[#333] transition-colors">
            <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
          </button>
        </div>
      </div>

      <!-- Right: Actions -->
      <div class="flex items-center gap-2">
        <button 
          id="create-button-upload" 
          @click="goUpload"
          class="p-2 hover:bg-gray-800 rounded-full"
          title="Create"
        >
          <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path></svg>
        </button>
        <div class="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center text-sm font-bold">U</div>
      </div>
    </nav>

    <div class="flex flex-1 overflow-hidden">
      <!-- Sidebar Navigation -->
      <aside class="w-64 bg-[#0F0F0F] hidden md:flex flex-col border-r border-gray-800 overflow-y-auto">
        <div class="p-3 space-y-1">
          <!-- Home (Current) -->
          <div class="flex items-center gap-5 px-3 py-2 bg-[#272727] rounded-xl cursor-default font-medium">
            <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z"/></svg>
            <span>Home</span>
          </div>
          
          <!-- Trending (Direct) -->
          <div 
            id="nav-trending" 
            @click="goTrending"
            class="flex items-center gap-5 px-3 py-2 hover:bg-[#272727] rounded-xl cursor-pointer transition-colors"
          >
            <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M17.66 11.2c-.23-.3-.51-.56-.77-.82-.67-.6-1.43-1.03-2.07-1.66C13.33 7.26 13 4.85 13.95 3c-.95.23-1.78.75-2.49 1.32-2.59 2.08-3.61 5.75-2.39 8.9.04.1.08.2.08.33 0 .22-.15.42-.35.5-.23.1-.47.04-.66-.12a.58.58 0 01-.14-.17c-1.13-1.43-1.31-3.48-.55-5.12C5.78 10 4.87 12.3 5 14.47c.06.5.12 1 .29 1.5.14.6.41 1.2.71 1.73 1.08 1.73 2.95 2.97 4.96 3.22 2.14.26 4.43-.26 6.07-1.75 1.88-1.71 2.51-4.48 1.63-6.97zm-4.56 5.8c-2.13.39-4.16-.68-4.88-2.68.66.42 1.48.56 2.24.43.3-.06.58-.16.85-.29.5-.23.95-.56 1.3-1 .23-.28.42-.59.62-.89.1-.15.2-.31.32-.45.02-.03.04-.05.07-.07.68 1.05 1.48 1.97 1.25 3.23-.21 1.15-1.01 1.41-1.77 1.72z"/></svg>
            <span>Trending</span>
          </div>

          <!-- Subscriptions (Hover Menu) -->
          <div 
            id="nav-subscriptions-menu" 
            class="relative group"
            @mouseenter="isSubscriptionsMenuOpen = true"
            @mouseleave="isSubscriptionsMenuOpen = false"
          >
            <div class="flex items-center gap-5 px-3 py-2 hover:bg-[#272727] rounded-xl cursor-pointer transition-colors">
              <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M18.7 8.7H5.3V7h13.4v1.7zm-1.7-5.3H7v1.8h10V3.4zm3.3 8.9v9.3H3.7v-9.3h16.6zM20 10.6H4v7.6h16v-7.6z"/></svg>
              <span>Subscriptions</span>
            </div>
            
            <!-- Hover Dropdown -->
            <div 
              v-if="isSubscriptionsMenuOpen"
              class="absolute left-full top-0 ml-2 w-48 bg-[#272727] rounded-xl shadow-xl border border-gray-700 p-2 z-50"
            >
              <div 
                class="menu-item-subscriptions px-3 py-2 hover:bg-gray-700 rounded-lg cursor-pointer"
                @click="goSubscriptions('subscriptions')"
              >
                All Subscriptions
              </div>
              <div 
                class="menu-item-library px-3 py-2 hover:bg-gray-700 rounded-lg cursor-pointer"
                @click="goSubscriptions('library')"
              >
                Library Shortcuts
              </div>
            </div>
          </div>

          <!-- Library (Click Toggle) -->
          <div class="relative">
            <div 
              id="nav-library-toggle"
              @click="toggleLibraryMenu"
              class="flex items-center gap-5 px-3 py-2 hover:bg-[#272727] rounded-xl cursor-pointer transition-colors"
              :class="{'bg-[#272727]': isLibraryMenuOpen}"
            >
              <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M4 6H2v14c0 1.1.9 2 2 2h14v-2H4V6zm16-4H8c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H8V4h12v12zM12 5.5v9l6-4.5z"/></svg>
              <span>Library</span>
            </div>

            <!-- Click Dropdown -->
            <div 
              v-if="isLibraryMenuOpen"
              class="ml-12 mt-1 space-y-1 border-l-2 border-gray-700 pl-2"
            >
              <div 
                id="nav-library-item-library"
                @click="goLibrary('library')"
                class="px-3 py-2 hover:bg-[#272727] rounded-lg cursor-pointer text-sm"
              >
                Main Library
              </div>
              <div 
                id="nav-library-item-history"
                @click="goLibrary('history')"
                class="px-3 py-2 hover:bg-[#272727] rounded-lg cursor-pointer text-sm"
              >
                History
              </div>
            </div>
          </div>
        </div>

        <div class="border-t border-gray-800 my-2 mx-4"></div>

        <div class="p-4 text-xs text-gray-500">
          <p>© 2024 YouTube Clone</p>
          <p class="mt-2">About Press Copyright</p>
          <p>Contact us Creators</p>
        </div>
      </aside>

      <!-- Main Content Area -->
      <main class="flex-1 overflow-y-auto bg-[#0F0F0F] relative">
        <!-- Hero Section with Background Image -->
        <div class="relative h-[400px] w-full group">
          <div class="absolute inset-0 z-0">
             <!-- Use ImageGetter directly in template via tool_call placeholder simulation -->
             <img src="/images/technology.jpg" alt="youtube channel art dark theme technology" class="w-full h-full object-cover opacity-60" />
             <div class="absolute inset-0 bg-gradient-to-t from-[#0F0F0F] to-transparent"></div>
          </div>
          
          <div class="relative z-10 h-full flex flex-col justify-end p-8 md:p-12 max-w-4xl">
            <h1 class="text-4xl md:text-6xl font-bold mb-4">Welcome to YouTube</h1>
            <p class="text-xl text-gray-200 mb-8 max-w-2xl">
              Discover content from the world's best creators. Watch, like, share, and upload your own videos.
            </p>
            <div class="flex flex-wrap gap-4">
               <button @click="goTrending" class="bg-[#FF0000] hover:bg-red-700 text-white px-8 py-3 rounded-full font-bold transition-colors flex items-center gap-2">
                 <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
                 Explore Trending
               </button>
               <button @click="goUpload" class="bg-white/10 hover:bg-white/20 text-white backdrop-blur px-8 py-3 rounded-full font-bold transition-colors">
                 Upload Video
               </button>
            </div>
          </div>
        </div>

        <!-- Decorative Video Grid (Visual Only - Actions routed to FSM paths) -->
        <div class="p-6">
          <h2 class="text-2xl font-bold mb-6">Recommended</h2>
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            <!-- Decorative Cards -->
            <div v-for="i in 8" :key="i" class="flex flex-col gap-2 group cursor-pointer opacity-75 hover:opacity-100 transition-opacity">
              <div class="aspect-video bg-gray-800 rounded-xl overflow-hidden relative">
                <div class="absolute inset-0 flex items-center justify-center text-gray-600">
                  <svg class="w-12 h-12" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
                </div>
              </div>
              <div class="flex gap-3">
                <div class="w-10 h-10 rounded-full bg-gray-700 flex-shrink-0"></div>
                <div>
                  <div class="h-4 w-40 bg-gray-800 rounded mb-2"></div>
                  <div class="h-3 w-24 bg-gray-800 rounded"></div>
                </div>
              </div>
            </div>
          </div>
          <div class="mt-12 text-center text-gray-500">
             <p>Sign in to see better recommendations</p>
          </div>
        </div>
      </main>
    </div>
  </div>
</template>

<script>
import { ref, onMounted } from 'vue'
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
    
    // UI State
    const isSubscriptionsMenuOpen = ref(false)
    const isLibraryMenuOpen = ref(false)

    const checkConsent = () => {
      if (!store.cookie_consent_given) {
        // In a real app, we might show a toast or shake the modal
        // But the modal is already covering the screen
        return false
      }
      return true
    }

    const refreshHome = () => {
      // Just for logo click
      if (checkConsent()) {
        router.push({ name: 'HOME' })
      }
    }

    const goTrending = () => {
      if (checkConsent()) {
        store.currentPageId = 'TRENDING'
        router.push({ name: 'TRENDING' })
      }
    }

    const goSubscriptions = (value) => {
      if (checkConsent()) {
        store.currentPageId = 'SUBSCRIPTIONS'
        // Logic for different values if needed, but FSM just says go to SUBSCRIPTIONS
        router.push({ name: 'SUBSCRIPTIONS' })
      }
    }

    const toggleLibraryMenu = () => {
      isLibraryMenuOpen.value = !isLibraryMenuOpen.value
    }

    const goLibrary = (value) => {
      if (checkConsent()) {
        store.currentPageId = 'LIBRARY'
        router.push({ name: 'LIBRARY' })
      }
    }

    const goUpload = () => {
      if (checkConsent()) {
        store.currentPageId = 'UPLOAD_VIDEO'
        router.push({ name: 'UPLOAD_VIDEO' })
      }
    }

    const goSearchResults = () => {
      if (checkConsent()) {
        store.currentPageId = 'SEARCH_RESULTS'
        router.push({ name: 'SEARCH_RESULTS' })
      }
    }

    return {
      store,
      isSubscriptionsMenuOpen,
      isLibraryMenuOpen,
      refreshHome,
      goTrending,
      goSubscriptions,
      toggleLibraryMenu,
      goLibrary,
      goUpload,
      goSearchResults
    }
  }
}
</script>