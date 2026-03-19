<template>
  <div class="fixed inset-0 z-[9999] flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
    <div class="bg-white rounded-2xl shadow-2xl w-full max-w-sm p-6 text-center">
      <div class="mx-auto bg-blue-100 w-16 h-16 flex items-center justify-center rounded-full mb-4 text-3xl">
        📍
      </div>
      <h3 class="text-xl font-bold text-gray-900 mb-2">Location Access Required</h3>
      <p class="text-gray-600 mb-6 text-sm">
        To show you products available for shipping to your region and accurate delivery estimates, please allow access to your location.
      </p>
      <div class="space-y-3">
        <button 
          id="permission-location-allow" 
          class="w-full py-3 px-4 text-white font-bold bg-blue-600 hover:bg-blue-700 rounded-xl shadow-lg transition-transform transform hover:scale-[1.02]"
          @click="$emit('allow')"
        >
          Allow Location Access
        </button>
        <button 
          id="permission-location-deny" 
          class="w-full py-2 px-4 text-gray-500 font-medium text-sm hover:text-gray-700 transition-colors"
          @click="$emit('deny')"
        >
          Not Now
        </button>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'PermissionModal',
  emits: ['allow', 'deny']
}
</script>