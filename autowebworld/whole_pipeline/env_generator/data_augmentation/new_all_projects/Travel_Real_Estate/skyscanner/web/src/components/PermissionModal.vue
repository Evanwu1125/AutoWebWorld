<template>
  <div v-if="isOpen" class="fixed inset-0 z-[9999] flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
    <div class="bg-white rounded-2xl shadow-2xl p-8 max-w-sm w-full text-center">
      <div class="bg-blue-50 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
      </div>
      <h3 class="text-xl font-bold text-gray-900 mb-2">Location Access</h3>
      <p class="text-gray-600 mb-8">
        This app needs access to your location to find the best flight deals near you.
      </p>
      <div class="flex flex-col gap-3">
        <button 
          id="permission-location-allow"
          @click="allowPermission"
          class="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-xl shadow-lg shadow-blue-600/20 transition-all"
        >
          Allow Location Access
        </button>
        <button 
          @click="denyPermission"
          class="w-full py-3 text-gray-500 font-medium hover:bg-gray-50 rounded-xl transition-colors"
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
    
    // Only show if we are on a page that requires it AND it's not granted yet
    // In FSM, FLIGHTS_SEARCH has this field.
    // Ideally pass a prop or check store context.
    const isOpen = computed(() => {
      return store.location_permission_granted === null && store.currentPageId === 'FLIGHTS_SEARCH'
    })
    
    const allowPermission = () => {
      store.location_permission_granted = true
    }
    
    const denyPermission = () => {
      // Optional handling
    }
    
    return {
      isOpen,
      allowPermission,
      denyPermission
    }
  }
}
</script>