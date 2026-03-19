<template>
  <div class="min-h-screen bg-[#0F0F0F] text-white flex flex-col items-center justify-center p-6 relative">
    <!-- Overlay Background -->
    <div class="absolute inset-0 bg-[url('/images/HeroBackground.jpg')] bg-cover opacity-20 blur-sm"></div>
    
    <div class="bg-[#1F1F1F] rounded-2xl p-8 max-w-md w-full shadow-2xl border border-gray-800 relative z-10">
      <h2 class="text-2xl font-bold mb-6 text-center">Confirm Subscription</h2>
      
      <p class="text-gray-300 mb-6 text-center leading-relaxed">
        Are you sure you want to subscribe to this channel? You'll receive updates in your subscription feed.
      </p>

      <div 
        id="notify-all-checkbox"
        @click="toggleNotify"
        class="flex items-center justify-center gap-3 mb-8 p-4 bg-[#272727] rounded-xl cursor-pointer hover:bg-[#333] transition-colors select-none border border-transparent"
        :class="{'border-blue-500': store.confirm_checked}"
      >
        <div class="w-6 h-6 rounded border border-gray-500 flex items-center justify-center" :class="{'bg-blue-500 border-blue-500': store.confirm_checked}">
          <svg v-if="store.confirm_checked" class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7"></path></svg>
        </div>
        <div class="flex items-center gap-2">
           <svg class="w-5 h-5 text-gray-400" fill="currentColor" viewBox="0 0 24 24"><path d="M12 22c1.1 0 2-.9 2-2h-4c0 1.1.9 2 2 2zm6-6v-5c0-3.07-1.63-5.64-4.5-6.32V4c0-.83-.67-1.5-1.5-1.5s-1.5.67-1.5 1.5v.68C7.64 5.36 6 7.92 6 11v5l-2 2v1h16v-1l-2-2zm-2 1H8v-6c0-2.48 1.51-4.5 4-4.5s4 2.02 4 4.5v6z"/></svg>
           <span class="font-medium">Turn on all notifications</span>
        </div>
      </div>

      <div class="flex gap-4">
        <button 
          id="subscribe-cancel" 
          @click="goBackWatch"
          class="flex-1 py-3 rounded-full font-medium hover:bg-white/10 transition-colors"
        >
          Cancel
        </button>
        <button 
          id="subscribe-confirm-button" 
          @click="confirmSubscribe"
          class="flex-1 bg-[#FF0000] hover:bg-red-600 text-white font-bold py-3 rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          :disabled="!store.confirm_checked"
        >
          Subscribe
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'CHANNEL_SUBSCRIBE_CONFIRM',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const toggleNotify = () => {
      store.confirm_checked = true
    }

    const confirmSubscribe = () => {
      if (store.confirm_checked) {
        store.currentPageId = 'SUBSCRIBE_SUCCESS'
        router.push({ name: 'SUBSCRIBE_SUCCESS' })
      }
    }

    const goBackWatch = () => {
      store.currentPageId = 'WATCH_VIDEO'
      router.push({ name: 'WATCH_VIDEO' })
    }

    return {
      store,
      toggleNotify,
      confirmSubscribe,
      goBackWatch
    }
  }
}
</script>