<template>
  <div v-if="show" class="fixed inset-0 z-[9999] flex items-center justify-center bg-black/50 backdrop-blur-sm transition-opacity duration-300">
    <div class="bg-white rounded-xl shadow-2xl p-6 max-w-md w-full mx-4 transform transition-all scale-100">
      <div class="flex flex-col items-center text-center">
        <div class="text-4xl mb-4">📍</div>
        <h2 class="text-xl font-bold text-gray-900 mb-2">Location Permission Required</h2>
        <p class="text-gray-600 mb-6 text-sm">
          This app needs access to your location to provide better service and personalized course recommendations.
        </p>
        <div class="flex gap-4 w-full">
          <button 
            id="permission-location-allow"
            @click="$emit('allow')" 
            class="flex-1 bg-blue-700 hover:bg-blue-800 text-white font-medium py-3 px-4 rounded-lg transition-colors"
          >
            Allow
          </button>
          <button 
            @click="$emit('deny')" 
            class="flex-1 bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium py-3 px-4 rounded-lg transition-colors"
          >
            Deny
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'PermissionModal',
  props: {
    show: {
      type: Boolean,
      default: false
    }
  },
  emits: ['allow', 'deny']
}
</script>