<template>
  <div class="fixed inset-0 z-[10000] flex items-end sm:items-center justify-center p-4 bg-black/50 backdrop-blur-sm transition-opacity duration-300">
    <div class="bg-white rounded-xl shadow-2xl w-full max-w-md p-6 transform transition-all scale-100">
      <div class="flex items-start space-x-4">
        <div class="flex-shrink-0 bg-orange-100 p-3 rounded-full">
          <span class="text-2xl">🍪</span>
        </div>
        <div class="flex-1">
          <h3 class="text-lg font-bold text-gray-900 mb-2">We Value Your Privacy</h3>
          <p class="text-sm text-gray-600 mb-6">
            We use cookies to enhance your shopping experience, serve personalized ads, and analyze our traffic. By clicking "Accept All", you consent to our use of cookies.
          </p>
          <div class="flex justify-end space-x-3">
            <button 
              id="cookie-decline" 
              class="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
              @click="$emit('decline')"
            >
              Decline
            </button>
            <button 
              id="cookie-accept" 
              class="px-6 py-2 text-sm font-bold text-white bg-red-600 hover:bg-red-700 rounded-lg shadow-lg hover:shadow-xl transition-all transform hover:-translate-y-0.5"
              @click="$emit('accept')"
            >
              Accept All
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
  emits: ['accept', 'decline']
}
</script>