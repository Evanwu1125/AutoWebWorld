<template>
  <div v-if="isVisible" class="fixed inset-0 z-[9999] flex items-center justify-center bg-black/60 backdrop-blur-md transition-opacity duration-300">
    <div class="bg-white rounded-2xl shadow-2xl p-8 max-w-sm w-full mx-4 text-center">
      <div class="mb-4 text-blue-500">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-12 w-12 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
      </div>
      <h3 class="text-xl font-bold text-gray-900 mb-2">Location Required</h3>
      <p class="text-gray-600 mb-8 text-sm">
        We need your location to show relevant local content in your feed.
      </p>
      <div class="flex flex-col gap-3">
        <button 
          id="permission-location-allow"
          @click="grantPermission"
          class="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-xl transition-colors shadow-lg"
        >
          Allow Location
        </button>
        <button 
          @click="close"
          class="w-full bg-gray-100 hover:bg-gray-200 text-gray-700 font-semibold py-3 px-6 rounded-xl transition-colors"
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
    
    // Only FEED page requires location in FSM (based on ACT_GRANT_LOCATION_FEED)
    const isVisible = computed(() => {
      return store.currentPageId === 'FEED' && store.location_permission_granted !== true
    })

    function grantPermission() {
      // FSM Action: ACT_GRANT_LOCATION_FEED
      store.location_permission_granted = true
    }
    
    function close() {
      // Optional: deny logic
    }

    return {
      isVisible,
      grantPermission,
      close
    }
  }
}
</script>