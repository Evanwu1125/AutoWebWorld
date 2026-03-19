<template>
  <div v-if="show" class="fixed inset-0 z-[10000] flex items-end justify-center sm:items-center p-4 bg-black/30 backdrop-blur-sm transition-opacity duration-300">
    <div class="bg-white rounded-xl shadow-2xl p-6 max-w-lg w-full animate-fade-in-up transform transition-all">
      <div class="flex items-start gap-4">
        <div class="flex-shrink-0 bg-blue-100 rounded-full p-2">
          <span class="text-2xl">🍪</span>
        </div>
        <div class="flex-1">
          <h3 class="text-lg font-semibold text-gray-900">We Value Your Privacy</h3>
          <p class="mt-2 text-sm text-gray-600 leading-relaxed">
            We use cookies to enhance your browsing experience, serve personalized content, and analyze our traffic. By clicking "Accept All", you consent to our use of cookies.
          </p>
        </div>
      </div>
      <div class="mt-6 flex flex-col sm:flex-row gap-3 sm:justify-end">
        <button type="button" 
          class="inline-flex justify-center rounded-md border border-gray-300 shadow-sm px-4 py-2 bg-white text-base font-medium text-gray-700 hover:bg-gray-50 focus:outline-none sm:text-sm">
          Customize
        </button>
        <button type="button" 
          id="cookie-accept"
          @click="accept"
          class="inline-flex justify-center rounded-md border border-transparent shadow-sm px-4 py-2 bg-blue-600 text-base font-medium text-white hover:bg-blue-700 focus:outline-none sm:text-sm">
          Accept All
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useSignatureStore } from '../stores/signature'
import { useRoute } from 'vue-router'

export default {
  name: 'CookieConsentModal',
  setup() {
    const signatureStore = useSignatureStore()
    const route = useRoute()

    const show = computed(() => {
      // Show on HOME page if not accepted
      return route.name === 'HOME' && signatureStore.cookie_accepted !== true
    })

    function accept() {
      signatureStore.cookie_accepted = true
    }

    return {
      show,
      accept
    }
  }
}
</script>