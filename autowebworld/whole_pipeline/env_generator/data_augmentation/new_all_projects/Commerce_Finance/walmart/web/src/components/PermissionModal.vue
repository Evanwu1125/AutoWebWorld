<template>
  <transition name="fade">
    <div v-if="isVisible" class="fixed inset-0 z-[9999] flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
      <div class="bg-white rounded-xl shadow-2xl max-w-sm w-full p-6 text-center transform transition-all">
        <div class="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
          <svg class="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
        </div>
        <h3 class="text-xl font-bold text-gray-900 mb-2">Location Required</h3>
        <p class="text-gray-600 mb-6">
          We need your location to show you the nearest store inventory and pricing.
        </p>
        <div class="flex flex-col gap-3">
          <button 
            id="permission-location-allow"
            class="w-full px-4 py-3 text-sm font-semibold text-white bg-blue-600 rounded-lg hover:bg-blue-700 transition-colors shadow-md"
          >
            Allow Location Access
          </button>
          <button 
            @click="deny"
            class="w-full px-4 py-3 text-sm font-semibold text-gray-600 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
          >
            Not Now
          </button>
        </div>
      </div>
    </div>
  </transition>
</template>

<script>
import { computed } from 'vue'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'PermissionModal',
  setup() {
    const store = useSignatureStore()
    
    // Check if current page requires permission (DEPARTMENTS page based on FSM)
    // FSM: DEPARTMENTS has signature_schema.location_permission_granted
    const isVisible = computed(() => {
      // Hardcoded check for pages that need it, or we could inspect the store state if we had schema metadata there.
      // Based on FSM, only DEPARTMENTS uses this.
      return store.currentPageId === 'DEPARTMENTS' && store.location_permission_granted === null
    })

    return {
      isVisible,
      deny: () => {} // Logic to handle deny if needed
    }
  }
}
</script>

<style scoped>
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>