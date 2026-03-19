<template>
  <div class="fixed inset-0 z-[10000] flex items-end sm:items-center justify-center bg-black/50 backdrop-blur-sm p-4">
    <div class="bg-black border border-[#2F3336] rounded-2xl shadow-2xl max-w-lg w-full p-6 transform transition-all">
      <div class="flex flex-col gap-4">
        <div class="flex items-start gap-4">
          <div class="text-4xl">🍪</div>
          <div class="flex-1">
            <h3 class="text-xl font-bold text-white mb-2">We Value Your Privacy</h3>
            <p class="text-[#71767B] text-sm leading-relaxed">
              We use cookies to enhance your browsing experience, serve personalized content, and analyze our traffic. By clicking 'Accept All', you consent to our use of cookies.
            </p>
          </div>
        </div>
        
        <div class="flex flex-col sm:flex-row gap-3 mt-4 justify-end">
          <button id="cookie-decline" class="px-6 py-2.5 rounded-full border border-[#536471] text-white font-bold hover:bg-white/10 transition-colors">
            Decline
          </button>
          <button id="cookie-accept" @click="$emit('accept')" class="px-6 py-2.5 rounded-full bg-white text-black font-bold hover:bg-[#EFF3F4] transition-colors">
            Accept All
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'CookieConsentModal',
  emits: ['accept']
}
</script>