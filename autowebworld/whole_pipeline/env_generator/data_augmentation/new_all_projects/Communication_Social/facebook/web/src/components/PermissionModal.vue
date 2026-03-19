<template>
  <div v-if="isVisible" class="fixed inset-0 z-[9999] flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
    <div class="bg-white rounded-xl shadow-2xl w-full max-w-md p-6 transform transition-all">
      <div class="text-center">
        <div class="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-blue-100 mb-4">
          <svg class="h-6 w-6 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
        </div>
        <h3 class="text-lg font-medium leading-6 text-gray-900 mb-2">Location Permission Required</h3>
        <p class="text-sm text-gray-500 mb-6">
          This feature needs access to your location to show relevant content and suggestions near you.
        </p>
        <div class="flex justify-center gap-3">
          <button 
            @click="denyPermission"
            class="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            Deny
          </button>
          <button 
            id="permission-location-allow"
            @click="allowPermission"
            class="px-4 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            Allow Location
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'PermissionModal',
  setup() {
    const signatureStore = useSignatureStore();
    
    // Pages that require location permission based on FSM schema
    const PAGES_REQUIRING_LOCATION = ['NEWS_FEED'];
    
    const isVisible = computed(() => {
      const currentPage = signatureStore.currentPageId;
      const requiresLocation = PAGES_REQUIRING_LOCATION.includes(currentPage);
      const notGranted = signatureStore.location_permission_granted === null;
      
      return requiresLocation && notGranted;
    });
    
    const allowPermission = () => {
      // Matches FSM action NEWS_FEED_LOCATION_PERMISSION_ALLOW
      signatureStore.location_permission_granted = true;
    };
    
    const denyPermission = () => {
      // Optional handling for deny
      console.log('Permission denied');
    };
    
    return {
      isVisible,
      allowPermission,
      denyPermission
    };
  }
}
</script>