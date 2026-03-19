<template>
  <div v-if="isVisible" class="fixed inset-0 z-[9999] flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
    <div class="bg-white rounded-xl shadow-2xl p-8 max-w-sm w-full text-center">
      <div class="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-blue-100 mb-4">
        <svg class="h-6 w-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"/>
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"/>
        </svg>
      </div>
      <h3 class="text-xl font-bold text-gray-900 mb-2">Location Permission Required</h3>
      <p class="text-sm text-gray-600 mb-6">
        This app needs access to your location to find providers near you and verify service eligibility.
      </p>
      <div class="space-y-3">
        <button
          id="permission-location-allow"
          @click="allowPermission"
          class="w-full bg-[#005DAA] text-white px-4 py-3 rounded-lg font-semibold hover:bg-[#004a87] transition-colors shadow-md"
        >
          Allow Location Access
        </button>
        <button
          id="permission-location-deny"
          @click="denyPermission"
          class="w-full bg-white border border-gray-300 text-gray-700 px-4 py-3 rounded-lg font-semibold hover:bg-gray-50 transition-colors"
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
    
    // Pages requiring location permission based on FSM schema
    const pagesRequiringPermission = ['LOGIN', 'VISIT_TYPE_SELECTION']

    const isVisible = computed(() => {
      const isRelevantPage = pagesRequiringPermission.includes(store.currentPageId)
      const notGranted = store.location_permission_granted === null || store.location_permission_granted === undefined
      return isRelevantPage && notGranted
    })

    const allowPermission = () => {
      // FSM Actions: ACT_LOGIN_GRANT_LOCATION, ACT_VT_GRANT_LOCATION
      store.location_permission_granted = true
    }

    const denyPermission = () => {
      // Optional
      store.location_permission_granted = false
    }

    return {
      isVisible,
      allowPermission,
      denyPermission
    }
  }
}
</script>