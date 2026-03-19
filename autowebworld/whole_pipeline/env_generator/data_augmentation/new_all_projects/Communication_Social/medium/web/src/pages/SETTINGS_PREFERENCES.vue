<template>
  <div class="min-h-screen bg-white">
    <nav class="border-b border-gray-200">
       <div class="max-w-3xl mx-auto px-4 h-16 flex items-center gap-4">
          <button id="settings-back-profile" @click="handleBackProfile" class="p-2 hover:bg-gray-100 rounded-full transition-colors">
             <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
               <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
             </svg>
          </button>
          <span class="font-sans font-bold text-lg">Settings</span>
       </div>
    </nav>

    <div class="max-w-3xl mx-auto px-4 py-12 space-y-12">
       <div>
          <h3 class="text-xl font-serif font-bold mb-6 border-b border-gray-100 pb-2">Preferences</h3>
          
          <div class="flex items-center justify-between py-4">
             <div>
                <div class="font-sans font-medium text-gray-900">Dark Mode</div>
                <div class="text-sm text-gray-500 font-sans">Use a dark theme for the application.</div>
             </div>
             <button 
                id="settings-dark-mode-toggle" 
                @click="toggleDarkMode" 
                :class="{
                   'w-12 h-6 rounded-full transition-colors relative': true,
                   'bg-green-600': darkMode,
                   'bg-gray-200': !darkMode
                }"
             >
                <span :class="{
                   'absolute top-1 w-4 h-4 rounded-full bg-white transition-all shadow-sm': true,
                   'left-7': darkMode,
                   'left-1': !darkMode
                }"></span>
             </button>
          </div>

          <div class="flex items-center justify-between py-4">
             <div>
                <div class="font-sans font-medium text-gray-900">Email Notifications</div>
                <div class="text-sm text-gray-500 font-sans">Receive emails about stories and recommendations.</div>
             </div>
             <button 
                id="settings-email-notifications-toggle" 
                @click="toggleEmail" 
                :class="{
                   'w-12 h-6 rounded-full transition-colors relative': true,
                   'bg-green-600': emailNotifs,
                   'bg-gray-200': !emailNotifs
                }"
             >
                <span :class="{
                   'absolute top-1 w-4 h-4 rounded-full bg-white transition-all shadow-sm': true,
                   'left-7': emailNotifs,
                   'left-1': !emailNotifs
                }"></span>
             </button>
          </div>
       </div>

       <div>
          <h3 class="text-xl font-serif font-bold mb-6 border-b border-gray-100 pb-2">Membership</h3>
          <div class="bg-yellow-50 p-6 rounded-lg border border-yellow-100 flex flex-col md:flex-row gap-6 items-start md:items-center justify-between">
             <div>
                <h4 class="font-bold font-serif text-lg mb-2">Become a Medium Member</h4>
                <p class="text-sm font-sans text-gray-700 mb-0">Get unlimited access to the best stories on Medium.</p>
             </div>
             <button 
                id="settings-membership-link" 
                @click="handleMembership" 
                class="bg-black text-white px-6 py-2 rounded-full font-sans font-medium hover:bg-gray-800 whitespace-nowrap"
             >
                Upgrade
             </button>
          </div>
       </div>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'SETTINGS_PREFERENCES',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    
    const darkMode = ref(false)
    const emailNotifs = ref(true)

    const toggleDarkMode = () => {
       darkMode.value = !darkMode.value
       signatureStore.dark_mode_enabled = darkMode.value
    }

    const toggleEmail = () => {
       emailNotifs.value = !emailNotifs.value
       signatureStore.email_notifications_enabled = emailNotifs.value
    }

    const handleMembership = async () => {
       signatureStore.setCurrentPageId('MEMBERSHIP_LANDING')
       await router.push({ name: 'MEMBERSHIP_LANDING' })
    }

    const handleBackProfile = async () => {
       signatureStore.setCurrentPageId('PROFILE_OVERVIEW')
       await router.push({ name: 'PROFILE_OVERVIEW' })
    }

    return {
       darkMode,
       emailNotifs,
       toggleDarkMode,
       toggleEmail,
       handleMembership,
       handleBackProfile
    }
  }
}
</script>