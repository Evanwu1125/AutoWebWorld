<template>
  <div id="app" class="min-h-screen bg-black text-[#E7E9EA] font-sans antialiased">
    <div class="flex justify-center max-w-[1265px] mx-auto h-screen overflow-hidden">
      <!-- Left Sidebar (Navigation) -->
      <header class="hidden sm:flex flex-col justify-between w-[88px] xl:w-[275px] px-2 h-full py-2 z-20 border-r border-[#2F3336]">
        <div class="flex flex-col gap-2 w-full xl:items-start items-center">
          <!-- Logo -->
          <router-link to="/" class="p-3 rounded-full hover:bg-white/10 w-fit transition-colors">
            <svg viewBox="0 0 24 24" aria-hidden="true" class="h-8 w-8 text-white fill-current"><g><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"></path></g></svg>
          </router-link>

          <!-- Navigation Links -->
          <nav class="flex flex-col gap-1 w-full mt-2">
            <div id="primary-home-link" @click="handleNav('HOME_TIMELINE')" class="group flex items-center xl:gap-4 p-3 xl:pr-8 rounded-full hover:bg-white/10 w-fit cursor-pointer transition-colors">
              <svg viewBox="0 0 24 24" aria-hidden="true" class="h-7 w-7 fill-current"><g><path d="M12 1.696L.622 8.807l1.06 1.696L3 9.679V19.5C3 20.881 4.119 22 5.5 22h13c1.381 0 2.5-1.119 2.5-2.5V9.679l1.318.824 1.06-1.696L12 1.696zM12 16.5c-1.933 0-3.5-1.567-3.5-3.5s1.567-3.5 3.5-3.5 3.5 1.567 3.5 3.5-1.567 3.5-3.5 3.5z"></path></g></svg>
              <span class="hidden xl:block text-xl font-medium">Home</span>
            </div>
            
            <div @click="handleNav('TRENDS_EXPLORE')" class="group flex items-center xl:gap-4 p-3 xl:pr-8 rounded-full hover:bg-white/10 w-fit cursor-pointer transition-colors">
              <svg viewBox="0 0 24 24" aria-hidden="true" class="h-7 w-7 fill-current"><g><path d="M10.25 3.75c-3.59 0-6.5 2.91-6.5 6.5s2.91 6.5 6.5 6.5c1.795 0 3.419-.726 4.596-1.904 1.178-1.177 1.904-2.801 1.904-4.596 0-3.59-2.91-6.5-6.5-6.5zm-8.5 6.5c0-4.694 3.806-8.5 8.5-8.5s8.5 3.806 8.5 8.5c0 1.986-.73 3.815-1.945 5.232l4.944 4.942-1.414 1.415-4.942-4.944C14.065 18.02 12.236 18.75 10.25 18.75c-4.694 0-8.5-3.806-8.5-8.5z"></path></g></svg>
              <span class="hidden xl:block text-xl font-medium">Explore</span>
            </div>

            <div id="nav-notifications" @click="handleNav('NOTIFICATIONS')" class="group flex items-center xl:gap-4 p-3 xl:pr-8 rounded-full hover:bg-white/10 w-fit cursor-pointer transition-colors">
              <svg viewBox="0 0 24 24" aria-hidden="true" class="h-7 w-7 fill-current"><g><path d="M12 1.75a8.25 8.25 0 00-8.25 8.25v2.887c0 2.225-1.077 4.192-2.81 5.378-.458.314-.492.969-.074 1.334.814.71 1.868 1.15 3.033 1.15H10.5a3.501 3.501 0 006.999 0h6.6c1.165 0 2.22-.44 3.034-1.15.418-.365.385-1.02-.073-1.334-1.734-1.186-2.811-3.153-2.811-5.378V10A8.25 8.25 0 0012 1.75zM14 22a1.5 1.5 0 11-3 0h3z"></path></g></svg>
              <span class="hidden xl:block text-xl font-medium">Notifications</span>
            </div>

            <div @click="handleNav('MESSAGES_INBOX')" class="group flex items-center xl:gap-4 p-3 xl:pr-8 rounded-full hover:bg-white/10 w-fit cursor-pointer transition-colors">
              <svg viewBox="0 0 24 24" aria-hidden="true" class="h-7 w-7 fill-current"><g><path d="M1.998 5.5c0-1.381 1.119-2.5 2.5-2.5h15c1.381 0 2.5 1.119 2.5 2.5v13c0 1.381-1.119 2.5-2.5 2.5h-15c-1.381 0-2.5-1.119-2.5-2.5v-13zm2.5-.5c-.276 0-.5.224-.5.5v2.764l8 3.638 8-3.636V5.5c0-.276-.224-.5-.5-.5h-15zm15.5 5.463l-8 3.636-8-3.638V18.5c0 .276.224.5.5.5h15c.276 0 .5-.224.5-.5v-8.037z"></path></g></svg>
              <span class="hidden xl:block text-xl font-medium">Messages</span>
            </div>

            <div @click="handleNav('BOOKMARKS')" class="group flex items-center xl:gap-4 p-3 xl:pr-8 rounded-full hover:bg-white/10 w-fit cursor-pointer transition-colors">
              <svg viewBox="0 0 24 24" aria-hidden="true" class="h-7 w-7 fill-current"><g><path d="M4 4.5C4 3.12 5.119 2 6.5 2h11C18.881 2 20 3.12 20 4.5v18.44l-8-5.71-8 5.71V4.5zM6.5 4c-.276 0-.5.22-.5.5v14.56l6-4.29 6 4.29V4.5c0-.28-.224-.5-.5-.5h-11z"></path></g></svg>
              <span class="hidden xl:block text-xl font-medium">Bookmarks</span>
            </div>

            <div id="nav-user-menu" class="group relative flex items-center xl:gap-4 p-3 xl:pr-8 rounded-full hover:bg-white/10 w-fit cursor-pointer transition-colors"
                 @mouseenter="hoverMenuOpen = true" @mouseleave="hoverMenuOpen = false">
              <svg viewBox="0 0 24 24" aria-hidden="true" class="h-7 w-7 fill-current"><g><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm5 11h-4v4h-2v-4H7v-2h4V7h2v4h4v2z"></path></g></svg>
              <span class="hidden xl:block text-xl font-medium">More</span>
              
              <!-- Hover Menu for Profile -->
              <div v-if="hoverMenuOpen" class="absolute bottom-full left-0 mb-2 w-48 bg-black border border-[#2F3336] rounded-xl shadow-lg overflow-hidden z-50">
                <div id="nav-user-menu-profile" @click.stop="handleNav('PROFILE_OVERVIEW')" class="px-4 py-3 hover:bg-white/10 cursor-pointer text-white font-medium">Profile</div>
                <div id="nav-user-menu-lists" class="px-4 py-3 hover:bg-white/10 cursor-pointer text-white font-medium">Lists</div>
                <div id="nav-user-menu-topics" class="px-4 py-3 hover:bg-white/10 cursor-pointer text-white font-medium">Topics</div>
                <div id="nav-user-menu-bookmarks" @click.stop="handleNav('BOOKMARKS')" class="px-4 py-3 hover:bg-white/10 cursor-pointer text-white font-medium">Bookmarks</div>
              </div>
            </div>

            <div id="primary-compose-tweet" @click="handleNav('COMPOSE_TWEET')" class="mt-4 bg-[#1D9BF0] hover:bg-[#1A8CD8] text-white rounded-full p-3 xl:px-8 xl:py-3 cursor-pointer shadow-lg transition-colors flex items-center justify-center">
              <svg viewBox="0 0 24 24" aria-hidden="true" class="h-6 w-6 xl:hidden fill-white"><g><path d="M23 3c-6.62-.1-10.38 2.421-13.05 6.03C7.29 12.61 6 17.331 6 22h2c0-1.007.07-2.012.19-3H12c4.1 0 7.48-3.082 7.94-7.054C22.79 10.147 23.17 6.359 23 3zm-7 8h-1.5v2H16c.63-.016 1.2-.08 1.72-.188C16.95 15.24 14.68 17 12 17H8.55c.57-2.512 1.57-4.851 3-6.78 2.16-2.912 5.29-4.911 9.45-5.187C20.95 8.079 19.9 11 16 11zM4 9V6H1V4h3V1h2v3h3v2H6v3H4z"></path></g></svg>
              <span class="hidden xl:block font-bold text-lg">Post</span>
            </div>
          </nav>
        </div>
      </header>

      <!-- Main Content -->
      <main class="flex-grow flex flex-col w-full sm:w-[600px] border-r border-[#2F3336] relative z-0 h-full overflow-y-auto hide-scrollbar">
        <router-view v-slot="{ Component }">
          <component :is="Component" />
        </router-view>
      </main>

      <!-- Right Sidebar (Widgets) -->
      <aside class="hidden lg:flex flex-col w-[350px] pl-8 py-4 h-full relative border-l border-[#2F3336]">
        <!-- Search -->
        <div class="sticky top-0 bg-black pb-4 z-10 w-full">
           <div class="relative group">
             <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 text-gray-500 group-focus-within:text-[#1D9BF0]"><g><path d="M10.25 3.75c-3.59 0-6.5 2.91-6.5 6.5s2.91 6.5 6.5 6.5c1.795 0 3.419-.726 4.596-1.904 1.178-1.177 1.904-2.801 1.904-4.596 0-3.59-2.91-6.5-6.5-6.5zm-8.5 6.5c0-4.694 3.806-8.5 8.5-8.5s8.5 3.806 8.5 8.5c0 1.986-.73 3.815-1.945 5.232l4.944 4.942-1.414 1.415-4.942-4.944C14.065 18.02 12.236 18.75 10.25 18.75c-4.694 0-8.5-3.806-8.5-8.5z"></path></g></svg>
             </div>
             <input type="text" placeholder="Search" class="w-full bg-[#202327] text-white rounded-full py-3 pl-12 pr-4 focus:outline-none focus:bg-black focus:ring-1 focus:ring-[#1D9BF0] border border-transparent focus:border-[#1D9BF0] placeholder-gray-500 text-sm">
           </div>
        </div>

        <!-- Trends Widget (Static for now) -->
        <div class="bg-[#16181C] rounded-2xl p-4 mb-4">
           <h2 class="font-bold text-xl mb-4 text-[#E7E9EA]">Trends for you</h2>
           <div class="space-y-4">
             <div class="cursor-pointer hover:bg-white/5 p-2 -mx-2 rounded-lg transition-colors" @click="handleNav('TRENDS_EXPLORE')">
               <div class="text-xs text-gray-500">Trending in US</div>
               <div class="font-bold">#VueJS</div>
               <div class="text-xs text-gray-500">25K posts</div>
             </div>
             <div class="cursor-pointer hover:bg-white/5 p-2 -mx-2 rounded-lg transition-colors" @click="handleNav('TRENDS_EXPLORE')">
               <div class="text-xs text-gray-500">Sports · Trending</div>
               <div class="font-bold">#NBAPlayoffs</div>
               <div class="text-xs text-gray-500">500K posts</div>
             </div>
           </div>
        </div>
      </aside>
    </div>

    <!-- Modals -->
    <CookieConsentModal v-if="showCookieConsent" @accept="handleAcceptCookies" />
    <PermissionModal v-if="showLocationPermission" title="Location Permission Required" message="This app needs access to your location to provide better service." @allow="handleAllowLocation" @deny="handleDenyLocation" />
    
    <!-- Mobile Bottom Nav -->
    <div class="sm:hidden fixed bottom-0 w-full bg-black border-t border-[#2F3336] flex justify-around py-3 z-50">
        <div @click="handleNav('HOME_TIMELINE')" class="cursor-pointer">
            <svg viewBox="0 0 24 24" aria-hidden="true" class="h-6 w-6 fill-current text-white"><g><path d="M12 1.696L.622 8.807l1.06 1.696L3 9.679V19.5C3 20.881 4.119 22 5.5 22h13c1.381 0 2.5-1.119 2.5-2.5V9.679l1.318.824 1.06-1.696L12 1.696zM12 16.5c-1.933 0-3.5-1.567-3.5-3.5s1.567-3.5 3.5-3.5 3.5 1.567 3.5 3.5-1.567 3.5-3.5 3.5z"></path></g></svg>
        </div>
        <div @click="handleNav('TRENDS_EXPLORE')" class="cursor-pointer">
            <svg viewBox="0 0 24 24" aria-hidden="true" class="h-6 w-6 fill-current text-gray-500"><g><path d="M10.25 3.75c-3.59 0-6.5 2.91-6.5 6.5s2.91 6.5 6.5 6.5c1.795 0 3.419-.726 4.596-1.904 1.178-1.177 1.904-2.801 1.904-4.596 0-3.59-2.91-6.5-6.5-6.5zm-8.5 6.5c0-4.694 3.806-8.5 8.5-8.5s8.5 3.806 8.5 8.5c0 1.986-.73 3.815-1.945 5.232l4.944 4.942-1.414 1.415-4.942-4.944C14.065 18.02 12.236 18.75 10.25 18.75c-4.694 0-8.5-3.806-8.5-8.5z"></path></g></svg>
        </div>
        <div @click="handleNav('NOTIFICATIONS')" class="cursor-pointer">
            <svg viewBox="0 0 24 24" aria-hidden="true" class="h-6 w-6 fill-current text-gray-500"><g><path d="M12 1.75a8.25 8.25 0 00-8.25 8.25v2.887c0 2.225-1.077 4.192-2.81 5.378-.458.314-.492.969-.074 1.334.814.71 1.868 1.15 3.033 1.15H10.5a3.501 3.501 0 006.999 0h6.6c1.165 0 2.22-.44 3.034-1.15.418-.365.385-1.02-.073-1.334-1.734-1.186-2.811-3.153-2.811-5.378V10A8.25 8.25 0 0012 1.75zM14 22a1.5 1.5 0 11-3 0h3z"></path></g></svg>
        </div>
        <div @click="handleNav('MESSAGES_INBOX')" class="cursor-pointer">
            <svg viewBox="0 0 24 24" aria-hidden="true" class="h-6 w-6 fill-current text-gray-500"><g><path d="M1.998 5.5c0-1.381 1.119-2.5 2.5-2.5h15c1.381 0 2.5 1.119 2.5 2.5v13c0 1.381-1.119 2.5-2.5 2.5h-15c-1.381 0-2.5-1.119-2.5-2.5v-13zm2.5-.5c-.276 0-.5.224-.5.5v2.764l8 3.638 8-3.636V5.5c0-.276-.224-.5-.5-.5h-15zm15.5 5.463l-8 3.636-8-3.638V18.5c0 .276.224.5.5.5h15c.276 0 .5-.224.5-.5v-8.037z"></path></g></svg>
        </div>
    </div>
  </div>
</template>

<script>
import { computed, ref, onMounted, watch } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import { useSignatureStore } from './stores/signature';
import CookieConsentModal from './components/CookieConsentModal.vue';
import PermissionModal from './components/PermissionModal.vue';
import fsmData from '../fsm.json';

export default {
  name: 'App',
  components: {
    CookieConsentModal,
    PermissionModal
  },
  setup() {
    const router = useRouter();
    const route = useRoute();
    const signatureStore = useSignatureStore();
    const hoverMenuOpen = ref(false);

    const handleNav = (pageName) => {
      signatureStore.setCurrentPageId(pageName);
      router.push({ name: pageName });
    };

    // Cookie Consent Logic
    const showCookieConsent = computed(() => {
      // Show on HOME page if consent not given
      return route.name === 'HOME' && signatureStore.cookie_consent_given !== true;
    });

    const handleAcceptCookies = () => {
      // In FSM, this action is ACT_HOME_ACCEPT_COOKIES
      signatureStore.cookie_consent_given = true;
    };

    // Location Permission Logic
    // Check FSM to see which pages need location permission
    const pagesWithLocation = fsmData.pages
      .filter(p => p.signature_schema && Object.prototype.hasOwnProperty.call(p.signature_schema, 'location_permission_granted'))
      .map(p => p.id);
    
    const showLocationPermission = computed(() => {
      return pagesWithLocation.includes(route.name) && signatureStore.location_permission_granted !== true;
    });

    const handleAllowLocation = () => {
      signatureStore.location_permission_granted = true;
    };

    const handleDenyLocation = () => {
       // Optional: handle denial
       // signatureStore.location_permission_granted = false; // If we had a denied state
    };

    return {
      handleNav,
      hoverMenuOpen,
      showCookieConsent,
      handleAcceptCookies,
      showLocationPermission,
      handleAllowLocation,
      handleDenyLocation
    };
  }
}
</script>

<style>
/* Custom Scrollbar hide */
.hide-scrollbar::-webkit-scrollbar {
  display: none;
}
.hide-scrollbar {
  -ms-overflow-style: none;
  scrollbar-width: none;
}
</style>