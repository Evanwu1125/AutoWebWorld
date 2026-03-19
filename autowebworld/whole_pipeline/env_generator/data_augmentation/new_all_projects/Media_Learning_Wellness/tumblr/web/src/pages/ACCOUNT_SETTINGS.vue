<template>
  <div class="min-h-screen bg-slate-900 text-white flex flex-col items-center py-10 px-4">
     <div class="w-full max-w-2xl">
       <!-- Header -->
       <div class="flex items-center gap-4 mb-8">
         <button 
           id="settings-back-dashboard" 
           @click="goDashboard"
           class="p-2 hover:bg-slate-800 rounded-full transition-colors text-slate-400"
         >
           <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" /></svg>
         </button>
         <h1 class="text-3xl font-bold">Account Settings</h1>
       </div>

       <div class="bg-slate-800 rounded-xl border border-slate-700 p-8 shadow-xl space-y-8">
          <!-- Display Name -->
          <div>
            <label class="block text-sm font-bold text-slate-400 uppercase mb-2">Display Name</label>
            <div 
              id="settings-display-name-input"
              @click="focusName"
              class="bg-slate-900 rounded-lg p-3 border border-slate-700 focus-within:border-blue-500 transition-colors"
            >
              <input 
                ref="nameInput"
                type="text" 
                placeholder="Your Blog Name"
                class="w-full bg-transparent outline-none text-white font-bold text-lg"
                :value="store.display_name"
                @input="handleNameInput"
              />
            </div>
          </div>

          <!-- Bio -->
          <div>
            <label class="block text-sm font-bold text-slate-400 uppercase mb-2">Bio</label>
            <div 
              id="settings-bio-textarea"
              @click="focusBio"
              class="bg-slate-900 rounded-lg p-3 border border-slate-700 focus-within:border-blue-500 transition-colors"
            >
              <textarea 
                ref="bioInput"
                placeholder="Tell us about yourself..."
                class="w-full h-32 bg-transparent outline-none text-slate-300 resize-none leading-relaxed"
                :value="store.bio"
                @input="handleBioInput"
              ></textarea>
            </div>
          </div>

          <!-- Theme Color Dropdown -->
          <div>
             <label class="block text-sm font-bold text-slate-400 uppercase mb-2">Theme Color</label>
             <div class="relative">
                <button 
                  id="settings-theme-color-dropdown"
                  @click="themeOpen = !themeOpen"
                  class="w-full bg-slate-900 rounded-lg p-3 border border-slate-700 flex justify-between items-center text-white"
                >
                  <span class="capitalize">{{ store.theme_color || 'Select Color' }}</span>
                  <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" /></svg>
                </button>
                
                <div v-if="themeOpen" class="absolute top-full left-0 mt-2 w-full bg-slate-800 rounded-lg shadow-xl border border-slate-700 overflow-hidden z-20">
                  <div id="theme-color-blue" @click="setTheme('blue')" class="px-4 py-3 hover:bg-slate-700 cursor-pointer flex items-center gap-2">
                    <div class="w-4 h-4 rounded-full bg-blue-500"></div> Blue
                  </div>
                  <div id="theme-color-dark" @click="setTheme('dark')" class="px-4 py-3 hover:bg-slate-700 cursor-pointer flex items-center gap-2">
                    <div class="w-4 h-4 rounded-full bg-slate-900 border border-slate-600"></div> Dark
                  </div>
                </div>
             </div>
          </div>

          <!-- Save Button -->
          <div class="pt-6 border-t border-slate-700">
             <button 
               id="settings-save-button" 
               @click="saveSettings"
               :disabled="!isValid"
               :class="[
                 'w-full py-4 rounded-lg font-bold text-lg transition-all transform hover:scale-[1.02]',
                 isValid ? 'bg-blue-500 text-white hover:bg-blue-600 shadow-lg shadow-blue-500/30' : 'bg-slate-700 text-slate-500 cursor-not-allowed'
               ]"
             >
               Save Changes
             </button>
          </div>
       </div>
     </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'ACCOUNT_SETTINGS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const nameInput = ref(null)
    const bioInput = ref(null)
    const themeOpen = ref(false)

    const focusName = () => nameInput.value?.focus()
    const focusBio = () => bioInput.value?.focus()

    const handleNameInput = (e) => store.display_name = e.target.value
    const handleBioInput = (e) => store.bio = e.target.value

    const setTheme = (color) => {
      store.theme_color = color
      themeOpen.value = false
    }

    const isValid = computed(() => {
      return store.display_name?.length > 0
    })

    const goDashboard = async () => {
      store.currentPageId = 'DASHBOARD_FEED'
      await router.push({ name: 'DASHBOARD_FEED' })
    }

    const saveSettings = async () => {
      if (!isValid.value) return
      store.success_message = "Profile updated successfully!"
      store.currentPageId = 'ACCOUNT_SETTINGS_SAVE_SUCCESS'
      await router.push({ name: 'ACCOUNT_SETTINGS_SAVE_SUCCESS' })
    }

    return {
      store,
      nameInput,
      bioInput,
      themeOpen,
      focusName,
      focusBio,
      handleNameInput,
      handleBioInput,
      setTheme,
      isValid,
      goDashboard,
      saveSettings
    }
  }
}
</script>