<template>
  <div class="flex h-screen bg-[#121212] text-white font-sans overflow-hidden">
    <aside class="w-64 bg-black flex-shrink-0 p-6 border-r border-[#282828] hidden md:block">
      <div id="back-home" @click="handleBackHome" class="flex items-center space-x-2 text-[#B3B3B3] hover:text-white cursor-pointer font-bold mb-8">
         <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M12.5 3.247a1 1 0 0 0-1 0L4 8.75v9a1 1 0 0 0 1 1h5v-5h4v5h5a1 1 0 0 0 1-1v-9l-7.5-5.503z"/></svg>
         <span>Home</span>
      </div>
      <div id="back-account-overview" @click="handleBackAccount" class="flex items-center space-x-2 text-[#B3B3B3] hover:text-white cursor-pointer font-bold">
         <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/></svg>
         <span>Account</span>
      </div>
    </aside>

    <main class="flex-1 overflow-y-auto p-8 md:p-12 max-w-4xl mx-auto w-full">
      <h1 class="text-3xl font-bold mb-8">Settings</h1>

      <!-- Theme -->
      <section class="mb-8">
         <h2 class="text-xl font-bold mb-4">Display</h2>
         <div class="bg-[#181818] p-6 rounded-lg flex items-center justify-between">
            <div>
               <div class="font-bold mb-1">Theme</div>
               <div class="text-[#B3B3B3] text-sm">Choose your preferred visual theme.</div>
            </div>
            <div id="settings-theme-dropdown" class="relative group">
               <button class="bg-[#282828] hover:bg-[#3E3E3E] text-white font-bold py-2 px-4 rounded-full flex items-center space-x-2">
                  <span>{{ currentTheme === 'light' ? 'Light' : 'Dark' }}</span>
                  <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
               </button>
               <div class="hidden group-hover:block absolute right-0 top-full mt-2 w-32 bg-[#282828] rounded shadow-xl z-50 border border-[#3E3E3E]">
                  <div id="settings-theme-dark" class="px-4 py-2 hover:bg-[#3E3E3E] cursor-pointer" @click="handleSelectTheme('dark')">Dark</div>
                  <div id="settings-theme-light" class="px-4 py-2 hover:bg-[#3E3E3E] cursor-pointer" @click="handleSelectTheme('light')">Light</div>
               </div>
            </div>
         </div>
      </section>

      <!-- Content -->
      <section class="mb-8">
         <h2 class="text-xl font-bold mb-4">Content</h2>
         <div class="bg-[#181818] p-6 rounded-lg flex items-center justify-between">
            <div>
               <div class="font-bold mb-1">Explicit Content</div>
               <div class="text-[#B3B3B3] text-sm">Allow playback of explicit-rated content.</div>
            </div>
            <div 
              id="settings-explicit-toggle" 
              @click="handleToggleExplicit" 
              class="w-12 h-6 rounded-full relative cursor-pointer transition-colors duration-300"
              :class="explicitEnabled ? 'bg-[#1DB954]' : 'bg-[#535353]'"
            >
               <div 
                 class="absolute top-1 left-1 bg-white w-4 h-4 rounded-full shadow-md transform transition-transform duration-300"
                 :class="{'translate-x-6': explicitEnabled}"
               ></div>
            </div>
         </div>
      </section>
    </main>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useSignatureStore } from '../stores/signature'
import { useRouter } from 'vue-router'

export default {
  name: 'SETTINGS',
  setup() {
    const store = useSignatureStore()
    const router = useRouter()

    const currentTheme = computed(() => store.selected_theme || 'dark')
    const explicitEnabled = computed(() => store.explicit_content_filter_enabled === true)

    const handleBackHome = async () => {
       store.setCurrentPageId('HOME')
       await router.push({ name: 'HOME' })
    }

    const handleBackAccount = async () => {
       store.setCurrentPageId('ACCOUNT_OVERVIEW')
       await router.push({ name: 'ACCOUNT_OVERVIEW' })
    }

    const handleSelectTheme = (theme) => {
       store.selected_theme = theme
    }

    const handleToggleExplicit = () => {
       store.explicit_content_filter_enabled = !store.explicit_content_filter_enabled
       // FSM effect says set to true, but toggle implies switch. FSM says 'click' and set true. 
       // Sticking to FSM logic strictly: set to true.
       store.explicit_content_filter_enabled = true
    }

    return {
       currentTheme,
       explicitEnabled,
       handleBackHome,
       handleBackAccount,
       handleSelectTheme,
       handleToggleExplicit
    }
  }
}
</script>