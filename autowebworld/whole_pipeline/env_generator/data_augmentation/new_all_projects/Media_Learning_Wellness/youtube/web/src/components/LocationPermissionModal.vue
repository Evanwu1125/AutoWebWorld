<template>
  <div v-if="isVisible" class="fixed inset-0 z-[9999] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
    <div class="bg-[#272727] text-white rounded-xl shadow-2xl max-w-md w-full border border-gray-700 p-6 text-center">
      <div class="w-16 h-16 bg-[#3EA6FF]/20 text-[#3EA6FF] rounded-full flex items-center justify-center mx-auto mb-4">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
      </div>
      <h2 class="text-xl font-bold mb-2">Location Access Required</h2>
      <p class="text-gray-300 mb-6">
        To show you trending videos in your area, this app needs access to your location.
      </p>
      <div class="flex gap-3 justify-center">
        <button 
          class="px-6 py-2 rounded-full border border-gray-600 text-gray-300 hover:bg-white/5 transition-colors"
          @click="isVisible = false"
        >
          Not Now
        </button>
        <button 
          id="permission-location-allow"
          @click="allowLocation"
          class="px-6 py-2 rounded-full bg-[#3EA6FF] text-black font-medium hover:bg-blue-400 transition-colors"
        >
          Allow Access
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'LocationPermissionModal',
  setup() {
    const store = useSignatureStore()
    
    // Show if permission is null (not yet granted)
    const isVisible = computed(() => store.location_permission_granted === null)
    
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