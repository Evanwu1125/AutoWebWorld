<template>
  <div v-if="isVisible" class="fixed inset-0 z-[10000] flex items-end sm:items-center justify-center bg-black/60 backdrop-blur-sm p-4">
    <div class="bg-white rounded-xl shadow-2xl p-6 max-w-lg w-full transform transition-all">
      <div class="flex items-start space-x-4">
        <div class="flex-shrink-0 pt-1">
          <span class="text-3xl">🍪</span>
        </div>
        <div class="flex-1">
          <h3 class="text-lg font-bold text-gray-900 mb-2">We Value Your Privacy</h3>
          <p class="text-sm text-gray-600 mb-4 leading-relaxed">
            We use cookies to enhance your browsing experience, serve personalized content, and analyze our traffic. By clicking 'Accept All', you consent to our use of cookies.
          </p>
          <div class="flex flex-col sm:flex-row gap-3">
            <button 
              id="cookie-accept"
              @click="$emit('accept')"
              class="flex-1 bg-gray-900 hover:bg-gray-800 text-white font-semibold py-2.5 px-4 rounded-lg transition-colors text-sm"
            >
              Accept All
            </button>
            <button 
              class="flex-1 bg-white border border-gray-300 hover:bg-gray-50 text-gray-700 font-semibold py-2.5 px-4 rounded-lg transition-colors text-sm"
            >
              Manage Preferences
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'CookieConsentModal',
  props: {
    isVisible: Boolean
  },
  emits: ['accept']
}
</script>