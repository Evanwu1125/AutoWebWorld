<template>
  <div class="min-h-screen flex flex-col bg-[#0d1117] text-white font-sans">
    <!-- Cookie Modal -->
    <CookieConsentModal />
    
    <!-- Navigation Bar -->
    <nav class="flex items-center justify-between px-6 py-4 bg-[#161b22] border-b border-gray-700">
      <div class="flex items-center space-x-4">
        <!-- Logo -->
        <div class="text-3xl font-bold tracking-tight text-white cursor-pointer hover:text-gray-300" id="github-logo">
          <svg height="32" aria-hidden="true" viewBox="0 0 16 16" version="1.1" width="32" data-view-component="true" class="octicon octicon-mark-github v-align-middle text-white fill-current">
            <path d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64-.18-1.32-.27-2-.27-.68 0-1.36.09-2 .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z"></path>
          </svg>
        </div>

        <!-- Menu Dropdown -->
        <div class="relative group">
           <button id="nav-dropdown-toggle" class="text-sm font-semibold text-white hover:text-gray-300 focus:outline-none">
             Menu ▾
           </button>
           <!-- Dropdown Content -->
           <div id="nav-dropdown-repos" 
                @click="navigateToRepos('menu')"
                class="absolute left-0 z-10 hidden w-48 mt-2 origin-top-left bg-[#161b22] border border-gray-700 rounded-md shadow-lg group-hover:block focus-within:block">
             <div class="py-1">
               <a href="#" class="block px-4 py-2 text-sm text-gray-300 hover:bg-gray-800 hover:text-white">Repositories</a>
             </div>
           </div>
        </div>

        <!-- Hover Menu -->
        <div id="nav-menu" class="relative group h-full flex items-center" @mouseenter="hoverMenuOpen = true" @mouseleave="hoverMenuOpen = false">
           <span class="text-sm font-semibold text-white cursor-pointer px-2">Explore</span>
           <div v-if="hoverMenuOpen" class="absolute top-full left-0 w-48 bg-[#161b22] border border-gray-700 rounded-md shadow-lg z-20">
             <div id="nav-menu-repos" @click="navigateToRepos('hover')" class="block px-4 py-2 text-sm text-gray-300 hover:bg-gray-800 hover:text-white cursor-pointer">
               Trending Repos
             </div>
           </div>
        </div>

        <!-- Direct Link -->
        <div id="nav-repos-direct" @click="navigateToRepos('direct')" class="text-sm font-semibold text-white cursor-pointer hover:text-gray-300">
          All Repos
        </div>
      </div>

      <div class="flex items-center space-x-4">
        <!-- User Profile & Menu -->
        <div class="relative group">
            <div id="user-dropdown-toggle" class="flex items-center space-x-2 cursor-pointer">
              <img src="/images/UserProfile.jpg" alt="Profile" class="w-8 h-8 rounded-full border border-gray-600" />
              <span class="text-sm font-semibold">▾</span>
            </div>
            <!-- User Dropdown -->
            <div class="absolute right-0 z-10 hidden w-48 mt-2 origin-top-right bg-[#161b22] border border-gray-700 rounded-md shadow-lg group-hover:block">
              <div class="py-1">
                <div id="user-dropdown-profile" @click="navigateToProfile('menu')" class="block px-4 py-2 text-sm text-gray-300 hover:bg-gray-800 hover:text-white cursor-pointer">
                  Your Profile
                </div>
              </div>
            </div>
        </div>

        <!-- User Hover Menu -->
        <div id="nav-user-menu" class="relative group h-full flex items-center" @mouseenter="userHoverMenuOpen = true" @mouseleave="userHoverMenuOpen = false">
          <span class="text-sm font-semibold text-white cursor-pointer px-2">Account</span>
          <div v-if="userHoverMenuOpen" class="absolute top-full right-0 w-48 bg-[#161b22] border border-gray-700 rounded-md shadow-lg z-20">
             <div id="nav-user-profile" @click="navigateToProfile('hover')" class="block px-4 py-2 text-sm text-gray-300 hover:bg-gray-800 hover:text-white cursor-pointer">
               My Profile
             </div>
          </div>
        </div>

        <!-- Direct Profile Link -->
        <div id="nav-profile-direct" @click="navigateToProfile('direct')" class="text-sm font-semibold text-white cursor-pointer hover:text-gray-300">
          Profile
        </div>
      </div>
    </nav>

    <!-- Hero Section -->
    <main class="flex-grow flex items-center justify-center relative overflow-hidden">
      <!-- Background Image using ImageGetter via manual img tag for now or style binding -->
      <div class="absolute inset-0 z-0">
        <img src="/images/GitHubUniverse.jpg" alt="GitHub Universe" class="w-full h-full object-cover opacity-20" />
      </div>
      
      <div class="relative z-10 max-w-4xl mx-auto text-center px-4">
        <h1 class="text-6xl font-extrabold tracking-tight mb-6 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500">
          Let's build from here
        </h1>
        <p class="text-2xl text-gray-400 mb-10 max-w-2xl mx-auto">
          The world's leading AI-powered developer platform.
        </p>
        <div class="flex flex-col sm:flex-row items-center justify-center gap-4">
           <button class="px-8 py-3 text-lg font-semibold rounded-md bg-white text-gray-900 hover:bg-gray-100 transition-colors">
             Sign Up for GitHub
           </button>
           <button class="px-8 py-3 text-lg font-semibold rounded-md border border-gray-500 hover:border-white transition-colors">
             Start a free enterprise trial
           </button>
        </div>
      </div>
    </main>

    <!-- Footer -->
    <footer class="bg-[#0d1117] border-t border-gray-800 py-8 px-6">
      <div class="max-w-6xl mx-auto flex flex-col md:flex-row justify-between items-center text-gray-500 text-xs">
        <div class="flex items-center space-x-4 mb-4 md:mb-0">
           <span>© 2025 GitHub, Inc.</span>
           <a href="#" class="hover:text-blue-400">Terms</a>
           <a href="#" class="hover:text-blue-400">Privacy</a>
           <a href="#" class="hover:text-blue-400">Security</a>
        </div>
        <div class="flex items-center space-x-6">
           <div class="w-5 h-5 bg-gray-700 rounded-full"></div>
           <div class="w-5 h-5 bg-gray-700 rounded-full"></div>
           <div class="w-5 h-5 bg-gray-700 rounded-full"></div>
        </div>
      </div>
    </footer>
  </div>
</template>

<script>
import { ref } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import CookieConsentModal from '../components/CookieConsentModal.vue';

export default {
  name: 'HOME',
  components: {
    CookieConsentModal
  },
  setup() {
    const router = useRouter();
    const store = useSignatureStore();
    const hoverMenuOpen = ref(false);
    const userHoverMenuOpen = ref(false);

    // Check permission helper
    const canNavigate = () => {
      // Based on FSM, navigation requires cookie consent
      return store.signature.cookie_consent_given === true;
    };

    const navigateToRepos = async (method) => {
      if (!canNavigate()) return;
      
      // In a real FSM engine, we'd fire the specific action ID
      // Here we just simulate the navigation effect
      store.setCurrentPageId('REPOSITORIES_LIST');
      await router.push({ name: 'REPOSITORIES_LIST' });
    };

    const navigateToProfile = async (method) => {
      if (!canNavigate()) return;
      store.setCurrentPageId('PROFILE_OVERVIEW');
      await router.push({ name: 'PROFILE_OVERVIEW' });
    };

    return {
      hoverMenuOpen,
      userHoverMenuOpen,
      navigateToRepos,
      navigateToProfile
    };
  }
}
</script>