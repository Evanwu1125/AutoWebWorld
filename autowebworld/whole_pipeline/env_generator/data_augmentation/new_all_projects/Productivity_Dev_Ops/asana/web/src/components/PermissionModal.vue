<template>
  <div v-if="isVisible" class="fixed inset-0 z-[9999] flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
    <div class="bg-white rounded-xl shadow-2xl max-w-sm w-full p-6 text-center animate-fade-in">
      <div class="w-16 h-16 bg-indigo-100 rounded-full flex items-center justify-center mx-auto mb-4">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8 text-indigo-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
      </div>
      <h2 class="text-xl font-bold text-gray-900 mb-2">Location Access</h2>
      <p class="text-gray-600 mb-6">
        This app needs access to your location to suggest local projects and team events near you.
      </p>
      <div class="flex flex-col gap-3">
        <button 
          id="permission-location-allow"
          @click="allow"
          class="w-full px-4 py-2 rounded-lg bg-indigo-600 text-white hover:bg-indigo-700 font-medium shadow-md transition-colors"
        >
          Allow Location Access
        </button>
        <button 
          id="permission-location-deny"
          @click="deny"
          class="w-full px-4 py-2 rounded-lg text-gray-500 hover:bg-gray-100 font-medium transition-colors"
        >
          Not Now
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'PermissionModal',
  setup() {
    const store = useSignatureStore()
    
    // FSM Logic: Only show on PROJECTS_LIST page if location_permission_granted is null
    const isVisible = computed(() => {
      return store.currentPageId === 'PROJECTS_LIST' && store.location_permission_granted === null
    })

    const allow = () => {
      // FSM Effect: set location_permission_granted = true
      store.location_permission_granted = true
    }

    const deny = () => {
      // Optional handling
      store.location_permission_granted = false
    }

    return {
      isVisible,
      allow,
      deny
    }
  }
}
</script>