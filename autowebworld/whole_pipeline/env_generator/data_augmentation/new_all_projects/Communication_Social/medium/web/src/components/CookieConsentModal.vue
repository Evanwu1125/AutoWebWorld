<template>
  <div v-if="visible" class="fixed inset-0 z-[9999] flex items-end justify-center pb-12 pointer-events-none">
    <div class="pointer-events-auto bg-white shadow-2xl rounded-lg p-6 w-full max-w-md border border-gray-200 mx-4 animate-fade-in-up">
      <div class="flex items-start space-x-4">
        <div class="text-3xl">🍪</div>
        <div class="flex-1">
          <h3 class="text-lg font-bold text-gray-900 mb-2">We Value Your Privacy</h3>
          <p class="text-sm text-gray-600 mb-6">
            We use cookies to enhance your browsing experience, serve personalized content, and analyze our traffic. By clicking 'Accept All', you consent to our use of cookies.
          </p>
          <div class="flex space-x-3">
            <button
              id="cookie-accept"
              @click="acceptCookies"
              class="flex-1 bg-black text-white px-4 py-2 rounded-full text-sm font-medium hover:bg-gray-800 transition-colors"
            >
              Accept All
            </button>
            <button
              id="cookie-decline"
              @click="declineCookies"
              class="px-4 py-2 text-gray-600 text-sm font-medium hover:text-gray-900 transition-colors"
            >
              Decline
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted, watch } from 'vue'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'CookieConsentModal',
  setup() {
    const signatureStore = useSignatureStore()
    const visible = ref(false)

    onMounted(() => {
      if (!signatureStore.cookie_consent_given) {
        visible.value = true
      }
    })

    watch(() => signatureStore.cookie_consent_given, (newVal) => {
      if (newVal) {
        visible.value = false
      }
    })

    const acceptCookies = () => {
      signatureStore.cookie_consent_given = true
    }

    const declineCookies = () => {
      signatureStore.cookie_consent_given = true
    }

    return {
      visible,
      acceptCookies,
      declineCookies
    }
  }
}
</script>

<style scoped>
.animate-fade-in-up {
  animation: fadeInUp 0.5s ease-out;
}

@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
</style>