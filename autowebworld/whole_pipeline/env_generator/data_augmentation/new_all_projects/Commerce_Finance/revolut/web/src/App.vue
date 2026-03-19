<template>
  <div id="app" class="h-full bg-gray-100 text-gray-900 font-sans antialiased">
    <router-view />
    
    <!-- Permission Modal -->
    <div v-if="showLocationPermission" class="fixed inset-0 z-[9999] flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div class="bg-white rounded-2xl shadow-2xl p-6 w-full max-w-sm mx-4 transform transition-all scale-100">
        <div class="flex justify-center mb-4">
          <div class="bg-blue-100 p-3 rounded-full">
            <svg class="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </div>
        </div>
        <h3 class="text-xl font-bold text-center text-gray-900 mb-2">Location Permission</h3>
        <p class="text-center text-gray-500 mb-6">We need your location to show relevant account services nearby.</p>
        <button 
          id="permission-location-allow"
          @click="grantLocationPermission"
          class="w-full py-3 px-4 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-xl transition-colors shadow-lg shadow-blue-200"
        >
          Allow Access
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { computed, watch } from 'vue'
import { useSignatureStore } from './stores/signature'
import { useRoute } from 'vue-router'

export default {
  name: 'App',
  setup() {
    const signatureStore = useSignatureStore()
    const route = useRoute()

    // Determine if we need to show location permission
    // Only for pages that have 'location_permission_granted' in their signature schema
    // In this FSM, it's ACCOUNTS_DASHBOARD
    const showLocationPermission = computed(() => {
      // Hardcoded check based on FSM analysis: ACCOUNTS_DASHBOARD has this requirement
      if (route.name === 'ACCOUNTS_DASHBOARD') {
        return signatureStore.location_permission_granted !== true
      }
      return false
    })

    const grantLocationPermission = () => {
      // This logic is actually handled by the FSM action, but for the modal button itself,
      // the FSM action "ACT_LOCATION_PERMISSION_ALLOW" is a click on #permission-location-allow
      // The actual store update happens in the page component's handler for this action,
      // but since this is a global modal, we might need to handle it here or ensure the click propagates.
      // However, FSM says action is on ACCOUNTS_DASHBOARD page. 
      // Ideally, the modal should be part of the page, but instructions say "Global Permission Modal".
      // We will let the click event bubble up or handle it via store if needed.
      // For now, let's assume the page component listens to this ID if it's mounted, 
      // OR we implement the store update here directly to satisfy the "click" effect.
      
      // Since the button ID matches the FSM action selector, we can implement the logic here 
      // effectively simulating the action effect.
      signatureStore.location_permission_granted = true
    }

    return {
      showLocationPermission,
      grantLocationPermission
    }
  }
}
</script>

<style>
/* Global styles */
body {
  @apply bg-gray-100;
}
</style>