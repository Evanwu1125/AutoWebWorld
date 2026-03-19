<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <!-- Navbar -->
    <header class="bg-white shadow-sm z-20">
      <div class="max-w-2xl mx-auto px-4 py-4 flex justify-between items-center">
        <div class="flex items-center gap-4">
          <button 
            id="back-home-from-settings" 
            @click="goHome" 
            class="hover:bg-gray-100 p-2 rounded-full transition"
          >
            <svg class="w-6 h-6 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
          </button>
          <h1 class="text-2xl font-bold text-gray-900">Settings</h1>
        </div>
      </div>
    </header>

    <!-- Main Content -->
    <main class="flex-1 max-w-2xl mx-auto w-full px-4 py-8">
      
      <div class="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden divide-y divide-gray-100">
        
        <!-- Theme Setting -->
        <div class="p-6 flex items-center justify-between">
          <div>
            <h3 class="text-lg font-medium text-gray-900">App Theme</h3>
            <p class="text-sm text-gray-500">Choose your preferred visual style</p>
          </div>
          <div class="relative">
            <button 
              id="theme-dropdown"
              @click="showThemeMenu = !showThemeMenu"
              class="flex items-center gap-2 bg-gray-50 hover:bg-gray-100 px-4 py-2 rounded-lg transition border border-gray-200"
            >
              <span>{{ currentTheme === 'dark' ? 'Dark Mode' : 'Light Mode' }}</span>
              <svg class="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
            </button>
            <div v-if="showThemeMenu" class="absolute right-0 mt-2 w-40 bg-white rounded-lg shadow-xl border border-gray-100 py-1 z-10">
              <div id="theme-light" @click="setTheme('light')" class="px-4 py-2 hover:bg-gray-50 cursor-pointer text-sm">Light Mode</div>
              <div id="theme-dark" @click="setTheme('dark')" class="px-4 py-2 hover:bg-gray-50 cursor-pointer text-sm">Dark Mode</div>
            </div>
          </div>
        </div>

        <!-- Sync Setting -->
        <div class="p-6 flex items-center justify-between">
          <div>
            <h3 class="text-lg font-medium text-gray-900">Cloud Sync</h3>
            <p class="text-sm text-gray-500">Keep your notes updated across devices</p>
          </div>
          <button 
            id="sync-toggle"
            @click="toggleSync"
            class="relative inline-flex h-6 w-11 items-center rounded-full transition-colors"
            :class="isSyncEnabled ? 'bg-purple-600' : 'bg-gray-200'"
          >
            <span 
              class="inline-block h-4 w-4 transform rounded-full bg-white transition-transform"
              :class="isSyncEnabled ? 'translate-x-6' : 'translate-x-1'"
            />
          </button>
        </div>

        <!-- Account / Sign Up Trigger -->
        <div class="p-6">
          <div class="bg-purple-50 rounded-xl p-6 flex flex-col items-center text-center">
            <h3 class="font-bold text-purple-900 mb-2">Create New Account</h3>
            <p class="text-sm text-purple-700 mb-4">Sign up to unlock advanced features and unlimited storage.</p>
            <button 
              id="open-sign-up-success"
              @click="goToSignUpSuccess"
              class="bg-purple-600 hover:bg-purple-700 text-white font-bold py-2 px-6 rounded-lg shadow transition-colors"
            >
              Complete Sign Up
            </button>
          </div>
        </div>

      </div>
    </main>
  </div>
</template>

<script>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'SETTINGS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    
    const showThemeMenu = ref(false)
    const currentTheme = ref(store.theme || 'light')
    const isSyncEnabled = ref(store.sync_enabled || false)

    const setTheme = (theme) => {
      currentTheme.value = theme
      store.theme = theme
      showThemeMenu.value = false
      // In a real app, this would apply class to body
    }

    const toggleSync = () => {
      isSyncEnabled.value = !isSyncEnabled.value
      store.sync_enabled = isSyncEnabled.value
    }

    const goToSignUpSuccess = async () => {
      store.setCurrentPageId('sign_up_new_account_success')
      await router.push({ name: 'sign_up_new_account_success' })
    }

    const goHome = async () => {
      store.setCurrentPageId('HOME')
      await router.push({ name: 'HOME' })
    }

    return {
      showThemeMenu,
      currentTheme,
      isSyncEnabled,
      setTheme,
      toggleSync,
      goToSignUpSuccess,
      goHome
    }
  }
}
</script>