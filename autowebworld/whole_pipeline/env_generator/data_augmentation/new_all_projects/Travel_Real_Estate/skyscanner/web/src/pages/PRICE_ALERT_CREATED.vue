<template>
  <div class="min-h-screen bg-slate-50 font-sans flex flex-col items-center justify-center p-6">
    <div class="bg-white rounded-3xl shadow-xl p-8 md:p-12 max-w-md w-full text-center space-y-8">
      <div class="w-20 h-20 bg-blue-100 rounded-full flex items-center justify-center mx-auto">
        <svg class="w-10 h-10 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>
      </div>

      <div>
        <h1 class="text-2xl font-bold text-gray-900 mb-2">Alert Created!</h1>
        <p class="text-gray-600">We'll let you know when the price changes.</p>
      </div>

      <div class="flex flex-col gap-4">
        <button 
          id="view-all-alerts"
          @click="goToAlertsList"
          class="w-full py-4 bg-[#0770E3] hover:bg-[#0660C3] text-white font-bold rounded-xl shadow-lg shadow-blue-600/20 transition-all transform hover:-translate-y-0.5"
        >
          View All Alerts
        </button>
        <button 
          id="alert-created-home"
          @click="goHome"
          class="w-full py-4 bg-white border-2 border-gray-100 hover:border-gray-300 text-gray-700 font-bold rounded-xl transition-all"
        >
          Back to Home
        </button>
        <div class="pt-4 border-t border-gray-100">
           <button id="back-alert-form" @click="goBack" class="text-sm text-gray-400 hover:text-gray-600 underline">Back (Debug)</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'PRICE_ALERT_CREATED',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const goToAlertsList = async () => {
      store.currentPageId = 'PRICE_ALERTS_LIST'
      await router.push({ name: 'PRICE_ALERTS_LIST' })
    }

    const goHome = async () => {
      store.currentPageId = 'HOME'
      await router.push({ name: 'HOME' })
    }

    const goBack = async () => {
      store.currentPageId = 'PRICE_ALERT_FORM'
      await router.push({ name: 'PRICE_ALERT_FORM' })
    }

    return {
      goToAlertsList,
      goHome,
      goBack
    }
  }
}
</script>