<template>
  <div class="h-screen bg-gray-50 flex flex-col items-center justify-center p-4">
    <div class="max-w-3xl w-full">
      <div class="flex justify-between items-center mb-8">
        <h1 class="text-3xl font-bold text-gray-900">Your Workspaces</h1>
        <button id="back-home" @click="handleBackHome" class="text-gray-500 hover:text-gray-900">Back to Home</button>
      </div>

      <div id="workspace-list" class="grid grid-cols-1 gap-4">
        <div 
          class="bg-white rounded-lg shadow-sm border border-gray-200 p-6 flex items-center justify-between hover:shadow-md transition cursor-pointer workspace-default"
          @click="handleOpenDefault"
        >
          <div class="flex items-center space-x-4">
            <div class="w-16 h-16 bg-purple-900 rounded-lg flex items-center justify-center text-white text-2xl font-bold">
              AC
            </div>
            <div>
              <h3 class="text-xl font-bold text-gray-900">Acme Corp</h3>
              <p class="text-gray-500">acme-corp.slack.com</p>
            </div>
          </div>
          <div class="text-purple-700 font-semibold">
            Launch
          </div>
        </div>

         <!-- User Profile Link -->
        <div class="mt-8 text-center">
            <button id="user-menu-profile" @click="handleOpenProfile" class="text-blue-600 hover:underline">
                View My Profile
            </button>
        </div>
      </div>
    </div>

    <!-- Location Permission Modal -->
    <div v-if="!signatureStore.location_permission_granted" class="fixed inset-0 bg-black/50 backdrop-blur-sm z-[9999] flex items-center justify-center p-4">
      <div class="bg-white rounded-xl shadow-2xl max-w-md w-full p-6 text-center transform transition-all scale-100">
        <div class="w-16 h-16 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center mx-auto mb-4">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
        </div>
        <h2 class="text-2xl font-bold text-gray-900 mb-2">Enable Location?</h2>
        <p class="text-gray-600 mb-6">We use your location to suggest relevant local channels and adjust timezones.</p>
        <button 
          id="permission-location-allow"
          @click="handleGrantLocation"
          class="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-lg transition"
        >
          Allow Location Access
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'WORKSPACE_OVERVIEW',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    function handleGrantLocation() {
      signatureStore.location_permission_granted = true
    }

    async function handleOpenDefault() {
      signatureStore.selected_workspace_id = 'ws_01' // Assuming ID gen logic handled here or store
      signatureStore.currentPageId = 'CHANNEL_LIST'
      await router.push({ name: 'CHANNEL_LIST' })
    }

    async function handleBackHome() {
      signatureStore.currentPageId = 'HOME'
      await router.push({ name: 'HOME' })
    }

    async function handleOpenProfile() {
      signatureStore.currentPageId = 'PROFILE_VIEW'
      await router.push({ name: 'PROFILE_VIEW' })
    }

    return {
      signatureStore,
      handleGrantLocation,
      handleOpenDefault,
      handleBackHome,
      handleOpenProfile
    }
  }
}
</script>