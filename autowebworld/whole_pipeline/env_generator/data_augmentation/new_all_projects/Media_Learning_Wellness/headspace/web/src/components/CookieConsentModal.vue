<template>
  <div class="fixed inset-0 z-[10000] flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm"
       aria-modal="true" role="dialog">
    <div class="bg-white rounded-2xl shadow-2xl max-w-md w-full p-6 animate-fade-in-up">
      <div class="flex items-start justify-between mb-4">
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 rounded-full bg-orange-100 flex items-center justify-center text-xl">
            🍪
          </div>
          <h2 class="text-xl font-bold text-gray-800">We Value Your Privacy</h2>
        </div>
      </div>
      
      <p class="text-gray-600 mb-6 leading-relaxed">
        We use cookies to enhance your browsing experience, serve personalized content, and analyze our traffic. 
        By clicking 'Accept All', you consent to our use of cookies to help you find your peace.
      </p>
      
      <div class="flex flex-col sm:flex-row gap-3">
        <button id="cookie-accept"
                @click="$emit('accept')"
                class="flex-1 bg-orange-500 hover:bg-orange-600 text-white font-semibold py-3 px-6 rounded-xl transition-colors duration-200 shadow-md hover:shadow-lg transform active:scale-95">
          Accept All
        </button>
        <button id="cookie-decline"
                @click="$emit('decline')"
                class="flex-1 bg-gray-100 hover:bg-gray-200 text-gray-700 font-semibold py-3 px-6 rounded-xl transition-colors duration-200">
          Decline
        </button>
      </div>
      
      <div class="mt-4 text-center">
        <a href="#" class="text-xs text-gray-400 hover:text-orange-500 underline">Privacy Policy</a>
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

<style scoped>
@keyframes fade-in-up {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
.animate-fade-in-up {
  animation: fade-in-up 0.4s ease-out forwards;
}
</style>