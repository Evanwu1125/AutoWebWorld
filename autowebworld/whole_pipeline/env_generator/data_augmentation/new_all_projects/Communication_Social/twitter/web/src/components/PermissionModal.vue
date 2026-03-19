<template>
  <div class="fixed inset-0 z-[9999] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
    <div class="bg-[#16181C] rounded-2xl shadow-2xl max-w-sm w-full p-6 text-center">
      <div class="w-16 h-16 bg-[#1D9BF0]/10 rounded-full flex items-center justify-center mx-auto mb-4">
        <svg viewBox="0 0 24 24" aria-hidden="true" class="h-8 w-8 text-[#1D9BF0] fill-current"><g><path d="M12 13c-2.209 0-4-1.791-4-4s1.791-4 4-4 4 1.791 4 4-1.791 4-4 4zm0-6c-1.105 0-2 .895-2 2s.895 2 2 2 2-.895 2-2-.895-2-2-2zm0 14.253c-1.391 0-5.83-2.673-8.293-8.156C2.693 10.61 2.5 8.353 2.5 7c0-5.238 4.262-9.5 9.5-9.5s9.5 4.262 9.5 9.5c0 1.353-.193 3.61-.986 6.097-2.036 6.38-7.14 8.156-8.514 8.156zM12 2C9.243 2 7 4.243 7 7c0 1.056.126 2.76.677 4.742 1.932 4.414 4.319 5.253 4.323 5.254.004-.001 2.39-.84 4.323-5.254C16.874 9.76 17 8.056 17 7c0-2.757-2.243-5-5-5z"></path></g></svg>
      </div>
      
      <h3 class="text-xl font-bold text-white mb-2">{{ title }}</h3>
      <p class="text-[#71767B] text-sm mb-6">
        {{ message }}
      </p>
      
      <div class="flex flex-col gap-3">
        <button id="permission-location-allow" @click="$emit('allow')" class="w-full py-3 rounded-full bg-[#1D9BF0] text-white font-bold hover:bg-[#1A8CD8] transition-colors">
          Allow
        </button>
        <button id="permission-location-deny" @click="$emit('deny')" class="w-full py-3 rounded-full border border-[#536471] text-white font-bold hover:bg-white/10 transition-colors">
          Don't Allow
        </button>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'PermissionModal',
  props: {
    title: {
      type: String,
      default: 'Permission Required'
    },
    message: {
      type: String,
      default: 'This app needs permission to continue.'
    }
  },
  emits: ['allow', 'deny']
}
</script>