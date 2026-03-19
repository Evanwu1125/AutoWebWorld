<template>
  <div v-if="show" class="fixed inset-0 z-[10000] flex items-center justify-center bg-black/50 backdrop-blur-sm transition-opacity duration-300">
    <div class="bg-white rounded-xl shadow-2xl p-6 max-w-md w-full mx-4 transform transition-all scale-100">
      <div class="flex flex-col items-center text-center">
        <div class="text-4xl mb-4">🍪</div>
        <h2 class="text-xl font-bold text-gray-900 mb-2">We Value Your Privacy</h2>
        <p class="text-gray-600 mb-6 text-sm">
          We use cookies to enhance your browsing experience, serve personalized content, and analyze our traffic. 
          By clicking 'Accept All', you consent to our use of cookies.
        </p>
        <button 
          id="cookie-accept"
          @click="$emit('accept')" 
          class="w-full bg-blue-700 hover:bg-blue-800 text-white font-medium py-3 px-4 rounded-lg transition-colors"
        >
          Accept All
        </button>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'CookieConsentModal',
  props: {
    show: {
      type: Boolean,
      default: false
    }
  },
  emits: ['accept']
}
</script>