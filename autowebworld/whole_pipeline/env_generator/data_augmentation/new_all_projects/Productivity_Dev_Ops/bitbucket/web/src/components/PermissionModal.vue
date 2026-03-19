<template>
  <div v-if="isVisible" class="fixed inset-0 z-[9999] flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
    <div class="bg-white rounded-xl shadow-xl max-w-sm w-full p-6 text-center">
      <div class="w-16 h-16 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center mx-auto mb-4">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
      </div>
      <h3 class="text-xl font-bold text-gray-900 mb-2">Location Permission</h3>
      <p class="text-gray-600 mb-6">
        This app needs access to your location to provide better service and relevant repository information.
      </p>
      <div class="space-y-3">
        <button 
          id="permission-location-allow"
          @click="allow"
          class="w-full px-4 py-2 text-white bg-blue-600 hover:bg-blue-700 rounded-lg shadow-sm font-medium transition-colors"
        >
          Allow Location Access
        </button>
        <button 
          id="permission-location-deny"
          @click="deny"
          class="w-full px-4 py-2 text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg font-medium transition-colors"
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
    const signatureStore = useSignatureStore()
    
    // Check if current page requires location permission and it's not granted yet
    // In FSM, REPO_LIST has location_permission_granted in signature_schema
    const isVisible = computed(() => {
      // Hardcoded check for pages that have this permission in schema based on FSM analysis
      // Pages: REPO_LIST
      const pagesWithPermission = ['REPO_LIST']
      return pagesWithPermission.includes(signatureStore.currentPageId) && 
             signatureStore.location_permission_granted !== true
    })

    function allow() {
      signatureStore.location_permission_granted = true
    }

    function deny() {
      // Optional: handle deny
    }

    return {
      isVisible,
      allow,
      deny
    }
  }
}
</script>