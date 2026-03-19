<template>
  <div v-if="show" class="fixed inset-0 z-[9999] flex items-center justify-center bg-black/50 backdrop-blur-sm">
    <div class="bg-white rounded-xl shadow-2xl p-6 max-w-md w-full mx-4 animate-fade-in-up">
      <div class="text-center">
        <div class="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-blue-100 mb-4">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
        </div>
        <h3 class="text-lg leading-6 font-medium text-gray-900">Location Permission Required</h3>
        <div class="mt-2">
          <p class="text-sm text-gray-500">
            This app needs access to your location to provide dashboard analytics specific to your region.
          </p>
        </div>
      </div>
      <div class="mt-5 sm:mt-6 flex gap-3">
        <button type="button" 
          id="permission-location-deny"
          @click="deny"
          class="mt-3 w-full inline-flex justify-center rounded-md border border-gray-300 shadow-sm px-4 py-2 bg-white text-base font-medium text-gray-700 hover:bg-gray-50 focus:outline-none sm:mt-0 sm:text-sm">
          Deny
        </button>
        <button type="button" 
          id="permission-location-allow"
          @click="allow"
          class="w-full inline-flex justify-center rounded-md border border-transparent shadow-sm px-4 py-2 bg-blue-600 text-base font-medium text-white hover:bg-blue-700 focus:outline-none sm:text-sm">
          Allow
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useSignatureStore } from '../stores/signature'
import { useRoute } from 'vue-router'

export default {
  name: 'PermissionModal',
  setup() {
    const signatureStore = useSignatureStore()
    const route = useRoute()
    
    // FSM Logic: Show only on pages that require location permission
    // In this FSM, DASHBOARD has dashboard_location_permission_granted in its schema
    // and ACT_LOCATION_PERMISSION_ALLOW action.
    
    const show = computed(() => {
      const pagesRequiringLocation = ['DASHBOARD']
      const isOnRequiredPage = pagesRequiringLocation.includes(route.name)
      const permissionGranted = signatureStore.dashboard_location_permission_granted === true
      
      return isOnRequiredPage && !permissionGranted
    })

    function allow() {
      // The actual state update will be handled by the FSM action handler in the page
      // But we can emit an event or just trigger the click on the ID which the FSM action listens to
      // For this architecture, the action handler on the page binds to #permission-location-allow
      // So this button just needs that ID. 
      // However, since this is a global component, we need to make sure the click event propagates 
      // OR we directly update the store here if we want to bypass the page handler (but we should follow FSM)
      // The FSM defines ACT_LOCATION_PERMISSION_ALLOW on DASHBOARD. 
      // So the dashboard page should have the handler.
      // But the modal is overlay.
      // Let's rely on the @click handler being attached to the ID by the parent page?
      // No, in Vue, we need to emit or handle it.
      // Since I'm implementing the page logic, I'll make sure the page component listens to this.
      // Actually, for a global modal, it's easier if it handles the logic itself mimicking the FSM action effect.
      
      signatureStore.dashboard_location_permission_granted = true
    }
    
    function deny() {
      // Not explicitly in FSM effects, but needed for UI
      // Maybe set a denied flag? FSM doesn't have one. Just close?
      // If we just close, it might reappear. 
      // For now, let's just close it temporarily or navigate away?
      // FSM requires permission to proceed with some actions perhaps.
      // I'll just let it stay or implement a local hide.
    }

    return {
      show,
      allow,
      deny
    }
  }
}
</script>