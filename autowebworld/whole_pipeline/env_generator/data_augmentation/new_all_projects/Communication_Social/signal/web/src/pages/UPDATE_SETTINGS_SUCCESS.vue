<template>
  <div class="h-screen bg-slate-50 flex flex-col items-center justify-center p-4">
    <div class="bg-white w-full max-w-sm rounded-2xl shadow-xl p-8 text-center space-y-6">
        <div class="w-20 h-20 bg-blue-100 text-blue-500 rounded-full flex items-center justify-center mx-auto">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-10 w-10" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
            </svg>
        </div>
        
        <div>
            <h2 class="text-2xl font-bold text-slate-800">Settings Saved</h2>
            <p class="text-slate-500 mt-2">Your preferences have been updated.</p>
        </div>

        <div class="space-y-3 pt-4">
            <button 
                id="settings-success-back-notifications" 
                @click="goBackNotifications"
                class="w-full py-3 px-4 bg-slate-100 text-slate-700 font-semibold rounded-xl hover:bg-slate-200 transition-colors"
            >
                Back to Notifications
            </button>
            <button 
                id="settings-success-go-home" 
                @click="goHome"
                class="w-full py-3 px-4 bg-blue-600 text-white font-semibold rounded-xl hover:bg-blue-700 shadow-md transition-colors"
            >
                Go Home
            </button>
        </div>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'UPDATE_SETTINGS_SUCCESS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const goBackNotifications = async () => {
        store.currentPageId = 'SETTINGS_NOTIFICATIONS'
        await router.push({ name: 'SETTINGS_NOTIFICATIONS' })
    }

    const goHome = async () => {
        store.currentPageId = 'HOME'
        await router.push({ name: 'HOME' })
    }

    return {
        goBackNotifications,
        goHome
    }
  }
}
</script>