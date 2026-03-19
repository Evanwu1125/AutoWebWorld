<template>
  <div class="min-h-screen bg-gray-100 flex flex-col">
    <CookieConsentModal />
    
    <!-- Navigation Bar -->
    <nav class="bg-white shadow-md z-20 sticky top-0">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between h-16">
          <div class="flex items-center gap-4">
            <div class="flex-shrink-0 flex items-center">
              <img class="h-10 w-10 text-blue-600" src="/images/Facebook.jpg" alt="Logo" />
            </div>
            <div class="hidden md:block">
              <div class="ml-4 flex items-baseline space-x-4">
                <button id="nav-news-feed" @click="goToNewsFeed" class="text-gray-600 hover:bg-gray-100 hover:text-gray-900 px-3 py-2 rounded-md text-sm font-medium transition-colors">
                  News Feed
                </button>
                <button id="nav-settings" @click="goToSettings" class="text-gray-600 hover:bg-gray-100 hover:text-gray-900 px-3 py-2 rounded-md text-sm font-medium transition-colors">
                  Settings
                </button>
              </div>
            </div>
          </div>
          
          <div class="flex items-center gap-4">
            <!-- Messenger Hover Menu -->
            <div class="relative group">
              <button id="nav-messenger" class="p-2 rounded-full bg-gray-200 hover:bg-gray-300 transition-colors">
                <svg class="h-6 w-6 text-black" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 2C6.48 2 2 6.03 2 11c0 2.87 1.56 5.47 3.99 7.18V22l3.63-1.99c1.39.38 2.88.38 4.27 0L17.5 22v-3.82c2.43-1.71 3.99-4.31 3.99-7.18 0-4.97-4.48-9-10-9zm-1.5 12.5l-2.75-2.92-5.25 2.92 5.75-6.08 2.75 2.92 5.25-2.92-5.75 6.08z"/>
                </svg>
              </button>
              <div class="absolute right-0 w-48 mt-2 bg-white rounded-md shadow-lg py-1 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-50">
                <button id="nav-messenger-inbox" @click="goToMessenger" class="block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100">
                  Inbox
                </button>
              </div>
            </div>

            <!-- Main Menu Dropdown -->
            <div class="relative">
              <button id="nav-menu" @click="toggleMenu" class="p-2 rounded-full bg-gray-200 hover:bg-gray-300 transition-colors">
                <svg class="h-6 w-6 text-black" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              </button>
              <div v-if="menuOpen" class="absolute right-0 mt-2 w-56 rounded-md shadow-lg bg-white ring-1 ring-black ring-opacity-5 z-50">
                <div class="py-1" role="menu">
                  <button id="nav-menu-friends" @click="goToFriends" class="block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100">
                    Friends
                  </button>
                  <button id="nav-menu-events" @click="goToEvents" class="block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100">
                    Events
                  </button>
                </div>
              </div>
            </div>
            
            <!-- User Avatar -->
            <img class="h-10 w-10 rounded-full cursor-pointer border border-gray-200" src="/images/photo1765160760.jpg" alt="User" />
          </div>
        </div>
      </div>
    </nav>

    <!-- Main Content (Hero Section) -->
    <main class="flex-grow flex items-center justify-center bg-cover bg-center relative" style="background-image: url('/images/Friends.jpg');">
      <div class="absolute inset-0 bg-black/40"></div>
      <div class="relative z-10 text-center px-4 max-w-4xl mx-auto">
        <h1 class="text-4xl md:text-6xl font-extrabold text-white mb-6 tracking-tight drop-shadow-lg">
          Connect with friends and the world around you.
        </h1>
        <p class="text-xl md:text-2xl text-gray-200 mb-10 drop-shadow-md">
          Share photos, updates, and stay in touch with the people who matter most.
        </p>
        <div class="flex flex-col sm:flex-row gap-4 justify-center">
          <button @click="goToNewsFeed" class="px-8 py-3 bg-blue-600 text-white font-bold rounded-full text-lg shadow-lg hover:bg-blue-700 transition-transform transform hover:-translate-y-1">
            See What's New
          </button>
          <button @click="goToFriends" class="px-8 py-3 bg-white text-blue-600 font-bold rounded-full text-lg shadow-lg hover:bg-gray-100 transition-transform transform hover:-translate-y-1">
            Find Friends
          </button>
        </div>
      </div>
    </main>
    
    <footer class="bg-white py-6 border-t border-gray-200">
      <div class="max-w-7xl mx-auto px-4 text-center text-gray-500 text-sm">
        &copy; 2025 Social Network Clone. All rights reserved.
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
    const signatureStore = useSignatureStore();
    const menuOpen = ref(false);

    const toggleMenu = () => {
      menuOpen.value = !menuOpen.value;
    };

    const goToNewsFeed = async () => {
      signatureStore.currentPageId = 'NEWS_FEED';
      await router.push({ name: 'NEWS_FEED' });
    };

    const goToFriends = async () => {
      signatureStore.currentPageId = 'FRIENDS_LIST';
      await router.push({ name: 'FRIENDS_LIST' });
    };

    const goToEvents = async () => {
      signatureStore.currentPageId = 'EVENTS_LIST';
      await router.push({ name: 'EVENTS_LIST' });
    };

    const goToMessenger = async () => {
      signatureStore.currentPageId = 'MESSENGER_INBOX';
      await router.push({ name: 'MESSENGER_INBOX' });
    };

    const goToSettings = async () => {
      signatureStore.currentPageId = 'SETTINGS_ACCOUNT';
      await router.push({ name: 'SETTINGS_ACCOUNT' });
    };

    return {
      menuOpen,
      toggleMenu,
      goToNewsFeed,
      goToFriends,
      goToEvents,
      goToMessenger,
      goToSettings
    };
  }
}
</script>