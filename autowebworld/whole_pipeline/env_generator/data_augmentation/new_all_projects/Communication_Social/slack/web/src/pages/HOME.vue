<template>
  <div class="h-screen flex flex-col items-center justify-center bg-purple-900 text-white relative overflow-hidden">
    <!-- Hero Background Image -->
    <div class="absolute inset-0 z-0 opacity-40">
      <img src="/images/Office.jpg" alt="Modern Office" class="w-full h-full object-cover" />
    </div>

    <div class="z-10 text-center max-w-2xl px-6">
      <h1 class="text-5xl font-bold mb-6 tracking-tight">Make work life simpler, more pleasant and more productive.</h1>
      
      <!-- Action: Go to Workspace (Direct) -->
      <button 
        id="enter-workspace-button"
        @click="handleEnterWorkspace"
        class="bg-white text-purple-900 font-bold py-4 px-8 rounded-md text-lg hover:shadow-xl transition-all mb-4 transform hover:scale-105"
      >
        Launch Workspace
      </button>

      <!-- Action: Go to Workspace (Hover Menu) -->
      <div id="workspace-menu" class="relative group inline-block ml-4">
        <button class="text-white border border-white/30 py-4 px-6 rounded-md hover:bg-white/10 transition">
          More Options
        </button>
        <div class="absolute hidden group-hover:block left-0 mt-2 w-48 bg-white text-gray-900 rounded-md shadow-xl py-2 text-left">
          <div id="workspace-menu .option-workspace" class="option-workspace px-4 py-2 hover:bg-gray-100 cursor-pointer" @click="handleEnterWorkspace">
            Open Workspace
          </div>
          <div class="option-help px-4 py-2 hover:bg-gray-100 cursor-pointer">
            Help Center
          </div>
        </div>
      </div>

      <!-- Action: Go to Workspace (Dropdown) -->
       <div class="relative inline-block ml-4">
         <button id="workspace-dropdown-toggle" @click="toggleDropdown" class="text-white underline opacity-80 hover:opacity-100">
           Sign In options
         </button>
         <div v-if="showDropdown" class="absolute left-0 mt-2 w-48 bg-white text-gray-900 rounded-md shadow-xl py-2 text-left z-50">
           <div id="workspace-dropdown-item" class="px-4 py-2 hover:bg-gray-100 cursor-pointer" @click="handleEnterWorkspace">
             Open Workspace
           </div>
           <div id="workspace-dropdown-signout" class="px-4 py-2 hover:bg-gray-100 cursor-pointer">
             Sign Out
           </div>
         </div>
       </div>
    </div>

    <!-- Cookie Consent Modal (Interceptor) -->
    <div v-if="!signatureStore.cookie_consent_given" class="fixed bottom-0 left-0 right-0 bg-white p-6 shadow-2xl z-[10000] border-t border-gray-200 transform transition-transform duration-300">
      <div class="max-w-4xl mx-auto flex flex-col md:flex-row items-center justify-between">
        <div class="mb-4 md:mb-0 pr-8">
          <h3 class="text-lg font-bold text-gray-900 mb-1">🍪 We value your privacy</h3>
          <p class="text-gray-600 text-sm">We use cookies to enhance your experience and analyze traffic. By clicking "Accept All", you consent to our use of cookies.</p>
        </div>
        <button 
          id="cookie-accept"
          @click="handleAcceptCookies"
          class="bg-green-700 hover:bg-green-800 text-white font-bold py-3 px-6 rounded-md transition whitespace-nowrap"
        >
          Accept All
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'HOME',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const showDropdown = ref(false)

    function handleAcceptCookies() {
      signatureStore.cookie_consent_given = true
    }

    async function handleEnterWorkspace() {
      if (signatureStore.cookie_consent_given) {
        signatureStore.currentPageId = 'WORKSPACE_OVERVIEW'
        await router.push({ name: 'WORKSPACE_OVERVIEW' })
      }
    }

    function toggleDropdown() {
      showDropdown.value = !showDropdown.value
    }

    return {
      signatureStore,
      handleAcceptCookies,
      handleEnterWorkspace,
      showDropdown,
      toggleDropdown
    }
  }
}
</script>