<template>
  <div v-if="show" class="fixed inset-0 z-[9999] flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
    <div class="bg-white rounded-xl shadow-2xl max-w-md w-full p-6 transform transition-all">
      <div class="flex items-center justify-center mb-4 text-blue-600">
        <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-map-pin"><path d="M20 10c0 6-9 13-9 13s-9-7-9-13a9 9 0 0 1 18 0Z"/><circle cx="12" cy="10" r="3"/></svg>
      </div>
      
      <h2 class="text-xl font-bold text-center text-gray-900 mb-2">Location Permission Required</h2>
      
      <p class="text-gray-600 text-center mb-6">
        This app needs access to your location to provide better service and find nearby meeting rooms.
      </p>
      
      <div class="flex flex-col gap-3">
        <button 
          id="permission-location-allow"
          @click="allowPermission"
          class="w-full py-2.5 px-4 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
        >
          Allow
        </button>
        
        <button 
          id="permission-location-deny"
          @click="denyPermission"
          class="w-full py-2.5 px-4 bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2"
        >
          Deny
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue';
import { useSignatureStore } from '../stores/signature';
import fsmData from '../../fsm.json';

export default {
  name: 'PermissionModal',
  setup() {
    const store = useSignatureStore();
    
    const show = computed(() => {
      // Check if current page requires location permission
      const currentPage = fsmData.pages.find(p => p.id === store.currentPageId);
      const requiresLocation = currentPage?.signature_schema && Object.prototype.hasOwnProperty.call(currentPage.signature_schema, 'location_permission_granted');
      
      // Show if required AND not yet granted/denied (null or false usually, but here we check strictly not true)
      return requiresLocation && store.location_permission_granted !== true;
    });

    const allowPermission = () => {
      // This matches the effect of ACT_DASHBOARD_ALLOW_LOCATION logic effectively
      // But strictly we should trigger the action if possible. 
      // For the modal generic logic, we update store directly or call action.
      // The FSM defines ACT_DASHBOARD_ALLOW_LOCATION specifically for DASHBOARD.
      // If multiple pages had this, we'd need dynamic action call.
      // For now, we'll try to invoke the action via store if it exists for current page.
      
      const actionId = `ACT_${store.currentPageId}_ALLOW_LOCATION`; // Convention guess?
      // Better: search actions for one that effects location_permission_granted
      
      store.handleAction('ACT_DASHBOARD_ALLOW_LOCATION'); // Direct call for DASHBOARD case
      // Or generic fallback:
      store.location_permission_granted = true;
    };

    const denyPermission = () => {
      // Optional logic
      // store.location_permission_granted = false; 
    };

    return {
      show,
      allowPermission,
      denyPermission
    };
  }
}
</script>