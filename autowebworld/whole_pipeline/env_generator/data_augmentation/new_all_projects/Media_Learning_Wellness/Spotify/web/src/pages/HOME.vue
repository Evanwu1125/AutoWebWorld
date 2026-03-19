<template>
  <div class="flex h-screen bg-black text-white font-sans overflow-hidden">
    <!-- Sidebar -->
    <aside class="w-64 bg-black flex-shrink-0 flex flex-col p-6 space-y-6">
      <div class="text-white mb-2 flex items-center space-x-2">
        <!-- Spotify Logo -->
        <svg viewBox="0 0 167.5 167.5" class="w-8 h-8 fill-current text-white"><path d="M83.7 0C37.5 0 0 37.5 0 83.7c0 46.3 37.5 83.7 83.7 83.7 46.3 0 83.7-37.5 83.7-83.7S130 0 83.7 0zM122 120.8c-1.4 2.5-4.6 3.2-7.1 1.7-19.8-12.1-44.8-14.9-74.2-8.1-2.8.6-5.6-1.1-6.2-3.9-.6-2.8 1.1-5.6 3.9-6.2 32-7.3 59.6-4.2 81.9 9.3 2.5 1.5 3.4 4.7 1.7 7.2zm10.1-22.5c-1.8 3-5.6 3.9-8.5 2.1-22.8-14-57.6-18.1-84.5-9.9-3.3 1-6.9-1-7.9-4.3-1-3.3 1-6.9 4.3-7.9 30.3-9.2 69.2-4.6 94.6 11 3 1.8 3.9 5.6 2 8.5zm.4-23c-27.3-16.2-72.3-17.7-98.4-9.7-4.2 1.3-8.6-1-9.9-5.2-1.3-4.2 1-8.6 5.2-9.9 30.3-9.2 79.7-7.4 111 11.2 3.8 2.2 5 7.1 2.8 10.9-2.2 3.9-7.2 5.1-10.7 2.7z"/></svg>
        <span class="text-2xl font-bold tracking-tight">Spotify</span>
      </div>
      
      <nav class="space-y-4">
        <div class="space-y-4">
          <a 
            id="nav-home" 
            class="flex items-center space-x-4 text-white font-bold opacity-100 cursor-default"
          >
            <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M12.5 3.247a1 1 0 0 0-1 0L4 8.75v9a1 1 0 0 0 1 1h5v-5h4v5h5a1 1 0 0 0 1-1v-9l-7.5-5.503z"/></svg>
            <span>Home</span>
          </a>
          
          <a 
            id="nav-browse-direct"
            class="flex items-center space-x-4 text-[#B3B3B3] hover:text-white transition-colors cursor-pointer font-bold"
            @click="handleGoToBrowse"
          >
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
            <span>Search / Browse</span>
          </a>

          <!-- Hover Menu for Library -->
          <div 
            id="nav-library-menu"
            class="group relative"
          >
            <a 
              class="flex items-center space-x-4 text-[#B3B3B3] hover:text-white transition-colors cursor-pointer font-bold py-2"
            >
              <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"/></svg>
              <span>Your Library</span>
            </a>
            
            <!-- Hover Options -->
            <div class="hidden group-hover:block absolute left-0 top-full mt-2 w-48 bg-[#282828] rounded-md shadow-xl z-50 overflow-hidden border border-[#3E3E3E]">
              <div 
                class="item-library px-4 py-3 hover:bg-[#3E3E3E] text-white cursor-pointer text-sm"
                @click="handleGoToLibrary('your_library')"
              >
                All Playlists
              </div>
              <div 
                class="item-recently-played px-4 py-3 hover:bg-[#3E3E3E] text-white cursor-pointer text-sm"
                @click="handleGoToLibrary('recently_played')"
              >
                Recently Played
              </div>
            </div>
          </div>
        </div>
        
        <div class="pt-6 border-t border-[#282828] space-y-4">
          <div class="flex items-center space-x-4 text-[#B3B3B3] hover:text-white cursor-pointer font-bold">
            <div class="bg-white p-1 rounded-sm"><svg class="w-4 h-4 text-black" fill="currentColor" viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg></div>
            <span>Create Playlist</span>
          </div>
          <div class="flex items-center space-x-4 text-[#B3B3B3] hover:text-white cursor-pointer font-bold">
            <div class="bg-gradient-to-br from-indigo-700 to-blue-300 p-1 rounded-sm opacity-70"><svg class="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 24 24"><path d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"/></svg></div>
            <span>Liked Songs</span>
          </div>
        </div>
      </nav>
    </aside>

    <!-- Main Content -->
    <main class="flex-1 flex flex-col relative bg-gradient-to-b from-[#202020] to-[#121212] overflow-y-auto">
      <!-- Top Bar -->
      <header class="h-16 flex items-center justify-end px-8 sticky top-0 bg-[#000000]/40 backdrop-blur-md z-20">
        <div class="flex items-center space-x-4">
          <!-- User/Signup Menus -->
          
          <!-- Signup Menu (Logged Out State Simulation) -->
          <div id="nav-user-toggle" class="relative group">
            <button class="text-[#B3B3B3] hover:text-white font-bold text-sm tracking-wider uppercase hover:scale-105 transition-transform">
              Sign up
            </button>
            <div class="hidden group-hover:block absolute right-0 top-full mt-2 w-32 bg-[#282828] rounded shadow-xl z-50">
               <div id="nav-signup" class="px-4 py-2 hover:bg-[#3E3E3E] text-white cursor-pointer" @click="handleGoToSignup">Sign up</div>
               <div id="nav-login" class="px-4 py-2 hover:bg-[#3E3E3E] text-white cursor-pointer">Log in</div>
            </div>
          </div>

          <!-- Account Menu -->
          <div id="nav-account-toggle" class="relative group ml-4">
            <button class="bg-white rounded-full p-2 hover:scale-105 transition-transform">
              <svg class="w-5 h-5 text-black" fill="currentColor" viewBox="0 0 24 24"><path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/></svg>
            </button>
            <div class="hidden group-hover:block absolute right-0 top-full mt-2 w-48 bg-[#282828] rounded shadow-xl z-50 border border-[#3E3E3E]">
              <div id="nav-account-overview" class="px-4 py-2 hover:bg-[#3E3E3E] text-white cursor-pointer text-sm" @click="handleGoToAccount('account_overview')">Account</div>
              <div id="nav-account-profile" class="px-4 py-2 hover:bg-[#3E3E3E] text-white cursor-pointer text-sm" @click="handleGoToAccount('profile')">Profile</div>
              <div id="nav-account-settings" class="px-4 py-2 hover:bg-[#3E3E3E] text-white cursor-pointer text-sm" @click="handleGoToAccount('settings')">Settings</div>
            </div>
          </div>
        </div>
      </header>

      <!-- Content -->
      <div class="p-8 pb-32">
        <div class="relative w-full h-80 rounded-lg overflow-hidden mb-12 group">
           <!-- Hero Image -->
           <img src="https://images.unsplash.com/photo-1493225255756-d9584f8606e9?q=80&w=2070&auto=format&fit=crop" alt="Music concert crowd" class="w-full h-full object-cover transform group-hover:scale-105 transition-transform duration-700" />
           <div class="absolute inset-0 bg-gradient-to-t from-[#121212] via-transparent to-transparent"></div>
           <div class="absolute bottom-8 left-8">
             <h1 class="text-5xl font-bold mb-4">Music for everyone.</h1>
             <p class="text-xl font-medium mb-6">Millions of songs. No credit card needed.</p>
             <button class="bg-[#1DB954] text-black font-bold py-3 px-8 rounded-full hover:scale-105 transition-transform uppercase tracking-widest text-sm">
               Get Spotify Free
             </button>
           </div>
        </div>

        <h2 class="text-2xl font-bold mb-6">Featured Playlists</h2>
        <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-6">
          <div v-for="i in 5" :key="i" class="bg-[#181818] p-4 rounded-md hover:bg-[#282828] transition-colors cursor-pointer group">
            <div class="relative aspect-square bg-[#333] mb-4 shadow-lg rounded-md overflow-hidden">
               <img :src="`/images/pl-${i}.jpg`" class="w-full h-full object-cover" />
               <div class="absolute right-2 bottom-2 bg-[#1DB954] rounded-full p-3 opacity-0 group-hover:opacity-100 transform translate-y-2 group-hover:translate-y-0 transition-all shadow-xl">
                 <svg class="w-6 h-6 text-black fill-current" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
               </div>
            </div>
            <h3 class="font-bold text-white mb-1 truncate">Playlist {{ i }}</h3>
            <p class="text-sm text-[#B3B3B3] line-clamp-2">Description of the playlist goes here...</p>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { useSignatureStore } from '../stores/signature'
import { useRouter } from 'vue-router'

export default {
  name: 'HOME',
  setup() {
    const store = useSignatureStore()
    const router = useRouter()

    const handleGoToBrowse = async () => {
      // Precondition check: cookie_consent_given == true
      if (store.cookie_consent_given === true) {
        store.setCurrentPageId('BROWSE')
        await router.push({ name: 'BROWSE' })
      }
    }

    const handleGoToLibrary = async (value) => {
      if (store.cookie_consent_given === true) {
        store.setCurrentPageId('YOUR_LIBRARY')
        await router.push({ name: 'YOUR_LIBRARY' })
      }
    }

    const handleGoToAccount = async (value) => {
      if (store.cookie_consent_given === true) {
        // Based on selection, but FSM simplifies all account menu items to go to ACCOUNT_OVERVIEW
        store.setCurrentPageId('ACCOUNT_OVERVIEW')
        await router.push({ name: 'ACCOUNT_OVERVIEW' })
      }
    }

    const handleGoToSignup = async () => {
      if (store.cookie_consent_given === true) {
        store.setCurrentPageId('SIGNUP')
        await router.push({ name: 'SIGNUP' })
      }
    }

    return {
      handleGoToBrowse,
      handleGoToLibrary,
      handleGoToAccount,
      handleGoToSignup
    }
  }
}
</script>