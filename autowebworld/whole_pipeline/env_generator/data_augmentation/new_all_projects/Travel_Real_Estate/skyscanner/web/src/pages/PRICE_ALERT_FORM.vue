<template>
  <div class="min-h-screen bg-slate-50 font-sans pb-12">
    <header class="bg-white shadow-sm sticky top-0 z-30">
      <div class="max-w-xl mx-auto px-6 h-16 flex items-center justify-between">
        <div id="back-flight-details-from-alert" @click="goBack" class="flex items-center gap-2 cursor-pointer text-[#002D5C] hover:text-blue-600 transition-colors">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"/></svg>
          <span class="font-medium">Back</span>
        </div>
        <div class="font-bold text-[#002D5C]">Create Price Alert</div>
        <div class="w-20"></div>
      </div>
    </header>

    <main class="max-w-xl mx-auto px-6 py-8 space-y-6">
      <div class="bg-blue-50 border border-blue-100 rounded-2xl p-6 text-center">
        <div class="w-16 h-16 bg-white rounded-full flex items-center justify-center mx-auto mb-4 shadow-sm">
           <svg class="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"/></svg>
        </div>
        <h3 class="font-bold text-[#002D5C] text-lg">Never miss a price drop</h3>
        <p class="text-blue-600/80 text-sm mt-1">We'll send you an email when the price for this flight changes.</p>
      </div>

      <div class="bg-white rounded-2xl shadow-sm border border-gray-100 p-8 space-y-6">
        <div>
          <label class="block text-sm font-bold text-gray-700 mb-2">Email Address</label>
          <input 
            id="alert-email-input"
            type="email" 
            @input="handleEmail"
            class="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none transition-all"
            placeholder="you@example.com"
          />
        </div>
        
        <div>
          <label class="block text-sm font-bold text-gray-700 mb-2">Alert Name (Optional)</label>
          <input 
            id="alert-name-input"
            type="text" 
            @input="handleName"
            class="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none transition-all"
            placeholder="e.g. Summer Trip"
          />
        </div>

        <button 
          id="alert-validate-button"
          @click="validateForm"
          class="w-full py-4 bg-gray-100 hover:bg-gray-200 text-gray-800 font-bold rounded-xl transition-colors"
        >
          Check Details
        </button>

        <button 
          v-if="isValid"
          id="create-alert-button"
          @click="submitAlert"
          class="w-full py-4 bg-[#0770E3] hover:bg-[#0660C3] text-white font-bold rounded-xl shadow-lg shadow-blue-600/20 transition-all transform hover:-translate-y-0.5"
        >
          Create Alert
        </button>
      </div>
    </main>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'PRICE_ALERT_FORM',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const isValid = computed(() => store.alert_form_valid)

    const handleEmail = () => store.alert_email_entered = true
    const handleName = () => store.alert_name_entered = true

    const validateForm = () => {
      if (store.alert_email_entered && store.alert_name_entered) {
        store.alert_form_valid = true
      }
    }

    const submitAlert = async () => {
      if (store.alert_form_valid) {
        store.currentPageId = 'PRICE_ALERT_CREATED'
        await router.push({ name: 'PRICE_ALERT_CREATED' })
      }
    }

    const goBack = async () => {
      store.currentPageId = 'FLIGHT_DETAILS'
      await router.push({ name: 'FLIGHT_DETAILS' })
    }

    return {
      isValid,
      handleEmail,
      handleName,
      validateForm,
      submitAlert,
      goBack
    }
  }
}
</script>