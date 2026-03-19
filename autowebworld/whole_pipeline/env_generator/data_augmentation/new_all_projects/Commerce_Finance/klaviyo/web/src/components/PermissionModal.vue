<template>
  <div v-if="isVisible" class="fixed inset-0 z-[9999] flex items-center justify-center bg-black/50 backdrop-blur-sm">
    <div class="bg-white rounded-xl shadow-2xl p-8 max-w-md w-full mx-4 text-center">
      <div class="mx-auto w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mb-4">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
      </div>
      <h3 class="text-xl font-bold text-gray-900 mb-2">Location Access Required</h3>
      <p class="text-gray-600 mb-6">We need your location to show you relevant campaign performance data in your region.</p>
      
      <button 
        id="permission-location-allow"
        @click="$emit('allow')"
        class="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-4 rounded-lg transition-colors"
      >
        Allow Location Access
      </button>
    </div>
  </div>
</template>

<script>
export default {
  name: 'PermissionModal',
  props: {
    isVisible: Boolean
  },
  emits: ['allow']
}
</script>