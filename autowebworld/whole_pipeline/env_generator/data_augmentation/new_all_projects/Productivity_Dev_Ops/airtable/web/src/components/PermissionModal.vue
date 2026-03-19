<template>
  <div v-if="isVisible" class="fixed inset-0 z-[9999] flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
    <div class="bg-white rounded-xl shadow-2xl max-w-sm w-full p-6 text-center">
      <div class="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
      </div>
      <h3 class="text-xl font-bold text-gray-900 mb-2">Location Access Required</h3>
      <p class="text-gray-600 text-sm mb-6">
        To provide location-based services for your bases, we need access to your current location.
      </p>
      <div class="space-y-3">
        <button 
          id="permission-location-allow"
          @click="handleAllow"
          class="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2.5 px-4 rounded-lg transition-colors"
        >
          Allow Access
        </button>
        <button 
          class="w-full bg-white hover:bg-gray-50 text-gray-500 font-medium py-2.5 px-4 rounded-lg transition-colors border border-gray-200"
        >
          Deny
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
    
    // Logic: Only show on BASES_DASHBOARD if permission not yet granted
    const isVisible = computed(() => {
      // Check FSM: ACT_BASES_GRANT_LOCATION exists on BASES_DASHBOARD
      // precondition: location_permission_granted == null
      return store.currentPageId === 'BASES_DASHBOARD' && store.location_permission_granted !== true
    })

    const handleAllow = () => {
      // Maps to ACT_BASES_GRANT_LOCATION
      store.location_permission_granted = true
    }

    return {
      isVisible,
      handleAllow
    }
  }
}
</script>