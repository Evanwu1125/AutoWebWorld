<template>
  <div v-if="isVisible" class="fixed inset-0 z-[9999] flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
    <div class="bg-white rounded-xl shadow-xl max-w-sm w-full p-6 text-center">
      <div class="mb-4 text-purple-600">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-12 w-12 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
      </div>
      <h3 class="text-xl font-bold text-gray-900 mb-2">Location Permission Required</h3>
      <p class="text-gray-600 mb-6">
        This app needs access to your location to provide better service and tag your notes with locations.
      </p>
      <button
        id="permission-location-allow"
        @click="allowLocation"
        class="w-full bg-purple-600 hover:bg-purple-700 text-white font-bold py-2 px-4 rounded-lg transition-colors"
      >
        Allow Location Access
      </button>
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

    // Check if current page requires permission based on signature schema
    // In this FSM, NOTEBOOK_LIST has 'location_permission_granted' in schema
    const isVisible = computed(() => {
      const needsPermission = store.current_page_id === 'NOTEBOOK_LIST'
      const notGranted = store.location_permission_granted !== true
      return needsPermission && notGranted
    })

    const allowLocation = () => {
      store.location_permission_granted = true
    }

    return {
      isVisible,
      allowLocation
    }
  }
}
</script>