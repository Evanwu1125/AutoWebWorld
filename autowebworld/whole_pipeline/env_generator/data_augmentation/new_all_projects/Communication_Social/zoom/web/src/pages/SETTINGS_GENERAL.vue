<template>
  <div class="min-h-screen bg-gray-50 p-8">
    <div class="max-w-4xl mx-auto bg-white rounded-xl shadow-lg overflow-hidden min-h-[600px] flex">
      <!-- Sidebar -->
      <div class="w-64 bg-gray-100 p-4 border-r border-gray-200">
        <h2 class="text-lg font-bold text-gray-700 mb-6 px-2">Settings</h2>
        <div class="space-y-1">
          <button class="w-full text-left px-4 py-2 bg-blue-100 text-blue-700 rounded-md font-medium">General</button>
          <button 
            id="settings-video-tab"
            @click="goToVideo"
            class="w-full text-left px-4 py-2 text-gray-600 hover:bg-gray-200 rounded-md font-medium"
          >
            Video
          </button>
        </div>
        
        <div class="mt-10 pt-10 border-t border-gray-200">
           <button 
            id="settings-general-back-profile" 
            @click="goBack"
            class="w-full text-left px-4 py-2 text-gray-500 hover:text-gray-800 flex items-center"
          >
            <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>
            Back to Profile
          </button>
        </div>
      </div>

      <!-- Content -->
      <div class="flex-1 p-8">
        <h1 class="text-2xl font-bold text-gray-900 mb-8">General Settings</h1>
        
        <div class="space-y-8">
          <!-- Language Selection -->
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">Language</label>
            <div class="relative w-64">
              <button 
                id="settings-language-dropdown"
                @click="toggleLang"
                class="w-full bg-white border border-gray-300 rounded-md px-4 py-2 text-left flex justify-between items-center hover:border-gray-400 focus:ring-2 focus:ring-blue-500"
              >
                <span>{{ currentLangLabel }}</span>
                <svg class="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
              </button>
              
              <div v-if="langOpen" class="absolute z-10 w-full bg-white border border-gray-300 rounded-md shadow-lg mt-1">
                <div 
                  id="settings-language-english" 
                  @click="selectLang('en')" 
                  class="px-4 py-2 hover:bg-blue-50 cursor-pointer"
                >English</div>
                <div 
                  id="settings-language-spanish" 
                  @click="selectLang('es')" 
                  class="px-4 py-2 hover:bg-blue-50 cursor-pointer"
                >Spanish</div>
                <div 
                  id="settings-language-french" 
                  @click="selectLang('fr')" 
                  class="px-4 py-2 hover:bg-blue-50 cursor-pointer"
                >French</div>
              </div>
            </div>
          </div>

          <!-- Theme Selection -->
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">Theme</label>
            <div class="relative w-64">
              <button 
                id="settings-theme-dropdown"
                @click="toggleTheme"
                class="w-full bg-white border border-gray-300 rounded-md px-4 py-2 text-left flex justify-between items-center hover:border-gray-400 focus:ring-2 focus:ring-blue-500"
              >
                <span>{{ currentThemeLabel }}</span>
                <svg class="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
              </button>
              
              <div v-if="themeOpen" class="absolute z-10 w-full bg-white border border-gray-300 rounded-md shadow-lg mt-1">
                <div 
                  id="settings-theme-light" 
                  @click="selectTheme('light')" 
                  class="px-4 py-2 hover:bg-blue-50 cursor-pointer"
                >Light</div>
                <div 
                  id="settings-theme-dark" 
                  @click="selectTheme('dark')" 
                  class="px-4 py-2 hover:bg-blue-50 cursor-pointer"
                >Dark</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'SETTINGS_GENERAL',
  setup() {
    const router = useRouter();
    const store = useSignatureStore();
    
    const langOpen = ref(false);
    const themeOpen = ref(false);

    const currentLangLabel = computed(() => {
      const map = { 'en': 'English', 'es': 'Spanish', 'fr': 'French' };
      return map[store.language] || 'English';
    });

    const currentThemeLabel = computed(() => {
      const map = { 'light': 'Light', 'dark': 'Dark' };
      return map[store.theme] || 'Light';
    });

    const toggleLang = () => langOpen.value = !langOpen.value;
    const toggleTheme = () => themeOpen.value = !themeOpen.value;

    const selectLang = (val) => {
      store.language = val;
      store.handleAction('ACT_SETTINGS_SELECT_LANGUAGE');
      langOpen.value = false;
    };

    const selectTheme = (val) => {
      store.theme = val;
      store.handleAction('ACT_SETTINGS_SELECT_THEME');
      themeOpen.value = false;
    };

    const goToVideo = async () => {
      if (store.handleAction('ACT_SETTINGS_GO_VIDEO')) {
        await router.push({ name: 'SETTINGS_VIDEO' });
      }
    };

    const goBack = async () => {
      if (store.handleAction('ACT_SETTINGS_GENERAL_BACK_PROFILE')) {
        await router.push({ name: 'PROFILE_OVERVIEW' });
      }
    };

    return {
      store,
      langOpen,
      themeOpen,
      currentLangLabel,
      currentThemeLabel,
      toggleLang,
      toggleTheme,
      selectLang,
      selectTheme,
      goToVideo,
      goBack
    };
  }
}
</script>