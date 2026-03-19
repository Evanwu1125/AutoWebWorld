<template>
  <div v-if="isVisible" class="fixed inset-0 z-[9999] flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
    <div class="bg-[#282828] rounded-xl shadow-xl max-w-sm w-full p-6 text-center border border-[#3E3E3E]">
      <div class="mb-4 text-[#1DB954]">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-12 w-12 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
      </div>
      <h3 class="text-xl font-bold text-white mb-2">Enable Location Services</h3>
      <p class="text-[#B3B3B3] mb-6 text-sm">
        To provide you with local concert recommendations and nearby events, we need access to your location.
      </p>
      <div class="flex flex-col space-y-3">
        <button 
          id="permission-location-allow"
          class="w-full bg-[#1DB954] hover:bg-[#1ed760] text-black font-bold py-3 px-4 rounded-full transition-transform hover:scale-105"
          @click="allowPermission"
        >
          Allow Location Access
        </button>
        <button 
          class="w-full bg-transparent border border-[#727272] text-white hover:border-white font-bold py-3 px-4 rounded-full transition-colors"
          @click="denyPermission"
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
    
    // Check if current page requires location permission and it's not granted yet
    const isVisible = computed(() => {
      const pagesRequiringLocation = ['BROWSE'] 
      return pagesRequiringLocation.includes(store.currentPageId) && 
             store.location_permission_granted === null
    })

    const allowPermission = () => {
      store.location_permission_granted = true
    }

    const denyPermission = () => {
      // Optional: Handle deny state
      // store.location_permission_granted = false 
    }

    return {
      isVisible,
      allowPermission,
      denyPermission
    }
  }
}
</script>