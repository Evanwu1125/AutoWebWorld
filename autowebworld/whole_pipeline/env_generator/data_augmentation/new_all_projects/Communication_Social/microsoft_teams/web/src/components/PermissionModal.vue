<template>
  <div v-if="show" class="fixed inset-0 z-[9999] flex items-center justify-center bg-black/50 backdrop-blur-sm">
    <div class="bg-white rounded-lg shadow-xl p-6 w-96 max-w-full m-4">
      <div class="text-center">
        <div class="bg-purple-100 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-4">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
        </div>
        <h3 class="text-lg font-semibold text-gray-900 mb-2">Location Permission Required</h3>
        <p class="text-gray-600 mb-6">This app needs access to your location to provide better service and find nearby teams.</p>
        <div class="flex flex-col gap-3">
          <button 
            id="permission-location-allow"
            @click="allow"
            class="w-full bg-[#6264A7] hover:bg-[#464775] text-white font-medium py-2 px-4 rounded transition-colors"
          >
            Allow
          </button>
          <button 
            @click="deny"
            class="w-full bg-white hover:bg-gray-50 text-gray-700 font-medium py-2 px-4 rounded border border-gray-300 transition-colors"
          >
            Deny
          </button>
        </div>
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
    const store = useSignatureStore()
    const route = useRoute()

    // Pages that require location permission based on FSM signature_schema
    // TEAMS_LIST, CALENDAR_VIEW, CHAT_LIST, CALLS_HUB have 'location_permission_granted'
    const locationPages = ['TEAMS_LIST', 'CALENDAR_VIEW', 'CHAT_LIST', 'CALLS_HUB']

    const show = computed(() => {
      // Check if current page requires permission
      if (!locationPages.includes(route.name)) return false;
      
      // Check if permission is not yet granted (null or false)
      return store.location_permission_granted !== true;
    })

    const allow = () => {
      store.location_permission_granted = true;
    }

    const deny = () => {
        // Optional deny handling
    }

    return {
      show,
      allow,
      deny
    }
  }
}
</script>