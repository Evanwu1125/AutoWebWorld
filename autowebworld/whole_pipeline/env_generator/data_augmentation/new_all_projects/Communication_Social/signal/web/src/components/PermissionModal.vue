<template>
  <div v-if="visible" class="fixed inset-0 z-[9999] flex items-center justify-center p-4 bg-slate-900/80 backdrop-blur-sm transition-opacity">
    <div class="bg-white rounded-xl shadow-2xl max-w-md w-full overflow-hidden">
      <div class="p-6 text-center">
        <div class="mx-auto bg-blue-100 w-16 h-16 rounded-full flex items-center justify-center mb-4">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
        </div>
        <h2 class="text-xl font-bold text-slate-800 mb-2">Location Permission Required</h2>
        <p class="text-slate-600 mb-6">
          This feature requires access to your location to find nearby users and groups securely.
        </p>
        <button 
          id="permission-location-allow"
          @click="grantLocation"
          class="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-4 rounded-lg transition-colors shadow-md"
        >
          Allow Location Access
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
    
    // Visible if current page is CHATS_LIST (from FSM definition) and permission is null
    const visible = computed(() => {
      // Check if current page requires location permission based on FSM schema
      // In FSM, CHATS_LIST has 'location_permission_granted' in schema
      const pagesWithLocation = ['CHATS_LIST'] 
      return pagesWithLocation.includes(store.currentPageId) && store.location_permission_granted === null
    })

    const grantLocation = () => {
      store.location_permission_granted = true
    }

    return {
      visible,
      grantLocation
    }
  }
}
</script>