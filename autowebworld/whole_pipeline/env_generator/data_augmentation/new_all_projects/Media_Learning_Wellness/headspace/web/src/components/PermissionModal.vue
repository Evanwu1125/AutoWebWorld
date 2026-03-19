<template>
  <div class="fixed inset-0 z-[9999] flex items-center justify-center p-4 bg-black/60 backdrop-blur-md"
       aria-modal="true" role="dialog">
    <div class="bg-white rounded-3xl shadow-2xl max-w-sm w-full p-8 text-center animate-bounce-in">
      <div class="w-16 h-16 mx-auto mb-6 bg-blue-100 rounded-full flex items-center justify-center text-blue-600">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
      </div>
      
      <h3 class="text-2xl font-bold text-gray-800 mb-3">Location Access</h3>
      <p class="text-gray-600 mb-8">
        We need your location to suggest meditation groups and events happening near you.
      </p>
      
      <div class="space-y-3">
        <button id="permission-location-allow"
                @click="$emit('allow')"
                class="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-xl transition-all duration-200 shadow-lg hover:shadow-blue-500/30 transform active:scale-95">
          Allow Location Access
        </button>
        <button id="permission-location-deny"
                @click="$emit('deny')"
                class="w-full bg-white hover:bg-gray-50 text-gray-500 font-semibold py-3 px-6 rounded-xl border border-gray-200 transition-colors">
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

<style scoped>
@keyframes bounce-in {
  0% { transform: scale(0.9); opacity: 0; }
  70% { transform: scale(1.05); opacity: 1; }
  100% { transform: scale(1); }
}
.animate-bounce-in {
  animation: bounce-in 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
}
</style>