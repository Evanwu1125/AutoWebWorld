<template>
  <div class="min-h-screen bg-slate-50 flex flex-col items-center justify-center p-4">
    <div class="w-full max-w-md bg-white rounded-2xl shadow-xl overflow-hidden">
      <!-- Hero Section with Background Image -->
      <div class="relative h-64 bg-blue-600">
        <img src="/images/SecureMessaging.jpg" alt="Secure Messaging" class="w-full h-full object-cover opacity-80" />
        <div class="absolute inset-0 bg-gradient-to-t from-blue-900/80 to-transparent flex flex-col justify-end p-6">
          <h1 class="text-3xl font-bold text-white mb-2">Signal Clone</h1>
          <p class="text-blue-100">Speak Freely. Private messaging for everyone.</p>
        </div>
      </div>

      <!-- Navigation Menu -->
      <div class="p-6 space-y-4">
        <h2 class="text-lg font-semibold text-slate-800 mb-4">Quick Actions</h2>
        
        <!-- Direct Navigation -->
        <button 
          id="nav-chats-direct"
          @click="goToChatsDirect"
          class="w-full flex items-center p-4 bg-slate-50 hover:bg-slate-100 rounded-xl transition-colors group"
        >
          <div class="w-12 h-12 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center mr-4 group-hover:bg-blue-600 group-hover:text-white transition-colors">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
            </svg>
          </div>
          <div class="text-left">
            <div class="font-semibold text-slate-900">Chats</div>
            <div class="text-sm text-slate-500">View your conversations</div>
          </div>
          <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-slate-400 ml-auto" viewBox="0 0 20 20" fill="currentColor">
            <path fill-rule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clip-rule="evenodd" />
          </svg>
        </button>

        <!-- Hover Navigation -->
        <div id="nav-main" class="w-full bg-slate-50 rounded-xl p-4" @mouseover="hoverNavMain">
          <button 
            id="nav-main-chats" 
            @click="goToChatsHover"
            class="w-full bg-purple-600 hover:bg-purple-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors"
          >
            Chats (Hover)
          </button>
        </div>

        <!-- Menu Navigation -->
        <div class="w-full bg-slate-50 rounded-xl p-4">
            <button 
              id="nav-menu-toggle" 
              @click="toggleMenu"
              class="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors mb-2"
            >
              Menu
            </button>
            <button 
              id="nav-menu-chats" 
              @click="goToChatsMenu"
              class="w-full bg-slate-200 hover:bg-slate-300 text-slate-900 font-semibold py-2 px-4 rounded-lg transition-colors"
            >
              Chats (Menu)
            </button>
        </div>

        <!-- Contacts -->
        <button 
          id="nav-contacts"
          @click="goToContacts"
          class="w-full flex items-center p-4 bg-slate-50 hover:bg-slate-100 rounded-xl transition-colors group"
        >
          <div class="w-12 h-12 bg-emerald-100 text-emerald-600 rounded-full flex items-center justify-center mr-4 group-hover:bg-emerald-600 group-hover:text-white transition-colors">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
            </svg>
          </div>
          <div class="text-left">
            <div class="font-semibold text-slate-900">Contacts</div>
            <div class="text-sm text-slate-500">Manage your contacts</div>
          </div>
          <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-slate-400 ml-auto" viewBox="0 0 20 20" fill="currentColor">
            <path fill-rule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clip-rule="evenodd" />
          </svg>
        </button>

        <!-- Settings -->
        <button 
          id="nav-settings"
          @click="goToSettings"
          class="w-full flex items-center p-4 bg-slate-50 hover:bg-slate-100 rounded-xl transition-colors group"
        >
          <div class="w-12 h-12 bg-slate-200 text-slate-600 rounded-full flex items-center justify-center mr-4 group-hover:bg-slate-600 group-hover:text-white transition-colors">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.356a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.356 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.356a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </div>
          <div class="text-left">
            <div class="font-semibold text-slate-900">Settings</div>
            <div class="text-sm text-slate-500">Privacy & Notifications</div>
          </div>
          <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-slate-400 ml-auto" viewBox="0 0 20 20" fill="currentColor">
            <path fill-rule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clip-rule="evenodd" />
          </svg>
        </button>

        <!-- Cookie Accept Button (Invisible on main UI, handled by Modal, but needs to exist for FSM selector if modal isn't enough) -->
        <!-- Actually, the modal uses #cookie-accept, so this is just a fallback or logic handler reference -->
        <button id="cookie-accept" class="hidden" @click="acceptCookies">Accept Cookies</button>
      </div>
      
      <div class="bg-slate-50 p-4 text-center text-xs text-slate-400">
        Signal Clone v1.0
      </div>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'HOME',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const goToChatsDirect = async () => {
      store.currentPageId = 'CHATS_LIST'
      await router.push({ name: 'CHATS_LIST' })
    }

    const goToChatsHover = async () => {
      store.currentPageId = 'CHATS_LIST'
      await router.push({ name: 'CHATS_LIST' })
    }

    const hoverNavMain = () => {
      // Logic for hover effect if needed
    }

    const toggleMenu = () => {
      // Logic for menu toggle
    }

    const goToChatsMenu = async () => {
      store.currentPageId = 'CHATS_LIST'
      await router.push({ name: 'CHATS_LIST' })
    }

    const goToContacts = async () => {
      store.currentPageId = 'CONTACTS_LIST'
      await router.push({ name: 'CONTACTS_LIST' })
    }

    const goToSettings = async () => {
      store.currentPageId = 'SETTINGS_PRIVACY'
      await router.push({ name: 'SETTINGS_PRIVACY' })
    }

    const acceptCookies = () => {
        store.cookie_consent_given = true
    }

    return {
      goToChatsDirect,
      goToChatsHover,
      hoverNavMain,
      toggleMenu,
      goToChatsMenu,
      goToContacts,
      goToSettings,
      acceptCookies
    }
  }
}
</script>