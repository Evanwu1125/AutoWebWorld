<template>
  <div class="min-h-screen bg-[#F1F2F2] font-sans">
    <!-- Navbar -->
    <nav class="bg-white shadow-sm sticky top-0 z-50">
      <div class="max-w-5xl mx-auto px-4 h-14 flex items-center justify-between">
        <div class="flex items-center space-x-6">
          <h1 class="text-[#B92B27] text-2xl font-bold font-serif tracking-tighter">Quora</h1>
          
          <!-- Navigation Items -->
          <div class="hidden md:flex items-center space-x-6">
            <button id="nav-feed-direct" @click="goFeed" class="text-[#B92B27] hover:bg-gray-50 px-3 py-1 rounded-full transition-colors">
              <span class="flex items-center gap-2">
                <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path></svg>
              </span>
            </button>
            
            <button id="nav-menu" class="text-gray-500 hover:text-gray-700 relative group">
              <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"></path></svg>
              
              <!-- Dropdown Menu -->
              <div class="absolute top-full left-0 mt-2 w-48 bg-white rounded-md shadow-lg py-1 hidden group-hover:block border border-gray-100">
                <div id="nav-menu-topics" @click="goTopics" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer">
                  Topics
                </div>
              </div>
            </button>
            
            <button class="text-gray-500 hover:text-gray-700">
              <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"></path></svg>
            </button>
            
            <button class="text-gray-500 hover:text-gray-700">
              <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"></path></svg>
            </button>
          </div>
        </div>

        <div class="flex items-center space-x-4">
          <div class="relative group" id="user-menu">
             <img src="/images/UserProfile.jpg" alt="Profile" class="w-8 h-8 rounded-full cursor-pointer object-cover border border-gray-200" />
             <!-- Hover Menu -->
             <div class="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg py-1 hidden group-hover:block border border-gray-100 z-50">
               <div id="user-menu-profile" @click="goProfile" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer">Profile</div>
               <div id="user-menu-settings" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer">Settings</div>
               <div id="user-menu-logout" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer">Logout</div>
             </div>
          </div>
          <button class="bg-[#B92B27] text-white px-4 py-1.5 rounded-full text-sm font-medium hover:bg-[#a02521] transition-colors">
            Add Question
          </button>
        </div>
      </div>
    </nav>

    <!-- Main Content -->
    <main class="max-w-5xl mx-auto px-4 py-8 grid grid-cols-1 md:grid-cols-12 gap-6">
      
      <!-- Left Sidebar -->
      <div class="hidden md:block md:col-span-2 space-y-2">
        <div class="bg-gray-100 p-2 rounded text-xs font-bold text-gray-500 uppercase tracking-wide">Feeds</div>
        <div class="flex items-center gap-2 p-2 hover:bg-gray-200 rounded cursor-pointer text-sm text-gray-700 bg-gray-200 font-medium">
          <span class="w-6 h-6 bg-white rounded flex items-center justify-center text-xs shadow-sm">Q</span>
          Quora
        </div>
        <div class="flex items-center gap-2 p-2 hover:bg-gray-200 rounded cursor-pointer text-sm text-gray-600">
          <span class="w-6 h-6 bg-white rounded flex items-center justify-center text-xs shadow-sm">T</span>
          Tech
        </div>
         <div class="flex items-center gap-2 p-2 hover:bg-gray-200 rounded cursor-pointer text-sm text-gray-600">
          <span class="w-6 h-6 bg-white rounded flex items-center justify-center text-xs shadow-sm">S</span>
          Science
        </div>
      </div>

      <!-- Center Feed -->
      <div class="md:col-span-7 space-y-4">
        <!-- Input Box Placeholder -->
        <div class="bg-white p-4 rounded-lg shadow-sm border border-gray-200">
          <div class="flex gap-3">
             <img src="/images/UserProfile.jpg" class="w-8 h-8 rounded-full object-cover" />
             <div class="flex-1 bg-gray-100 rounded-full px-4 py-1.5 text-gray-500 text-sm cursor-pointer hover:bg-gray-200 transition-colors">
               What do you want to ask or share?
             </div>
          </div>
          <div class="flex justify-between mt-3 px-8 text-gray-500">
             <div class="flex items-center gap-1 cursor-pointer hover:bg-gray-50 px-2 py-1 rounded">
               <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
               <span class="text-sm font-medium">Ask</span>
             </div>
             <div class="flex items-center gap-1 cursor-pointer hover:bg-gray-50 px-2 py-1 rounded">
               <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z"></path></svg>
               <span class="text-sm font-medium">Answer</span>
             </div>
             <div class="flex items-center gap-1 cursor-pointer hover:bg-gray-50 px-2 py-1 rounded">
               <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"></path></svg>
               <span class="text-sm font-medium">Post</span>
             </div>
          </div>
        </div>

        <!-- Empty State / Welcome -->
        <div class="bg-white p-6 rounded-lg shadow-sm border border-gray-200 text-center py-12">
          <img src="/images/Welcome.jpg" alt="Welcome" class="w-full h-48 object-cover rounded-lg mb-6" />
          <h2 class="text-2xl font-bold text-gray-900 mb-2">Welcome to your Feed</h2>
          <p class="text-gray-600 mb-6">Follow topics and ask questions to see more content here.</p>
          <button @click="goFeed" class="bg-blue-600 text-white px-6 py-2 rounded-full font-medium hover:bg-blue-700 transition-colors">
            Go to Main Feed
          </button>
        </div>
      </div>

      <!-- Right Sidebar -->
      <div class="hidden md:block md:col-span-3 space-y-4">
         <div class="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
           <h3 class="font-bold text-gray-700 mb-3 border-b border-gray-100 pb-2">Improve Your Feed</h3>
           <div class="space-y-3">
             <div class="flex items-center gap-2 text-sm text-gray-600">
               <input type="checkbox" checked class="text-blue-600 rounded" />
               <span>Visit your feed</span>
             </div>
             <div class="flex items-center gap-2 text-sm text-gray-600">
               <input type="checkbox" class="text-blue-600 rounded" />
               <span>Follow 5 more topics</span>
             </div>
             <div class="flex items-center gap-2 text-sm text-gray-600">
               <input type="checkbox" class="text-blue-600 rounded" />
               <span>Upvote 3 good answers</span>
             </div>
             <div class="flex items-center gap-2 text-sm text-gray-600">
               <input type="checkbox" class="text-blue-600 rounded" />
               <span>Ask your first question</span>
             </div>
           </div>
         </div>
      </div>
    </main>

    <!-- Global Components -->
    <CookieConsentModal />
    <PermissionModal />
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import CookieConsentModal from '../components/CookieConsentModal.vue'
import PermissionModal from '../components/PermissionModal.vue'

export default {
  name: 'HOME',
  components: {
    CookieConsentModal,
    PermissionModal
  },
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    async function goFeed() {
      store.currentPageId = 'FEED'
      await router.push({ name: 'FEED' })
    }

    async function goTopics() {
      store.currentPageId = 'TOPIC_LIST'
      await router.push({ name: 'TOPIC_LIST' })
    }

    async function goProfile() {
      store.currentPageId = 'PROFILE'
      await router.push({ name: 'PROFILE' })
    }

    return {
      goFeed,
      goTopics,
      goProfile
    }
  }
}
</script>