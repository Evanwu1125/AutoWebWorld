<template>
  <div class="h-screen bg-slate-50 flex flex-col">
    <header class="bg-white shadow-sm z-20 px-4 py-3 flex items-center justify-between">
        <div class="flex items-center">
            <button id="settings-privacy-back-home" @click="goHome" class="p-2 text-slate-500 hover:text-blue-600 mr-4">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
                </svg>
            </button>
            <h1 class="text-xl font-bold text-slate-800">Privacy</h1>
        </div>
    </header>

    <div class="flex-1 overflow-y-auto p-4">
        <div class="max-w-md mx-auto space-y-6">
            <div class="bg-white rounded-2xl shadow-sm overflow-hidden divide-y divide-slate-100">
                <!-- Read Receipts -->
                <div class="p-4 flex items-center justify-between">
                    <div>
                        <h3 class="font-medium text-slate-800">Read Receipts</h3>
                        <p class="text-sm text-slate-500">See when messages are read</p>
                    </div>
                    <button 
                        id="toggle-read-receipts" 
                        @click="toggleReadReceipts"
                        :class="['w-12 h-6 rounded-full p-1 transition-colors', readReceipts ? 'bg-blue-600' : 'bg-slate-300']"
                    >
                        <div :class="['bg-white w-4 h-4 rounded-full shadow-md transform transition-transform', readReceipts ? 'translate-x-6' : 'translate-x-0']"></div>
                    </button>
                </div>

                <!-- Typing Indicators -->
                <div class="p-4 flex items-center justify-between">
                    <div>
                        <h3 class="font-medium text-slate-800">Typing Indicators</h3>
                        <p class="text-sm text-slate-500">See when others are typing</p>
                    </div>
                    <button 
                        id="toggle-typing-indicators" 
                        @click="toggleTyping"
                        :class="['w-12 h-6 rounded-full p-1 transition-colors', typingIndicators ? 'bg-blue-600' : 'bg-slate-300']"
                    >
                        <div :class="['bg-white w-4 h-4 rounded-full shadow-md transform transition-transform', typingIndicators ? 'translate-x-6' : 'translate-x-0']"></div>
                    </button>
                </div>
            </div>

            <div class="bg-white rounded-2xl shadow-sm overflow-hidden">
                <button 
                    id="settings-nav-notifications" 
                    @click="goToNotifications"
                    class="w-full p-4 flex items-center justify-between hover:bg-slate-50 transition-colors"
                >
                    <span class="font-medium text-slate-800">Notifications</span>
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
                    </svg>
                </button>
            </div>
        </div>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'SETTINGS_PRIVACY',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const readReceipts = ref(false)
    const typingIndicators = ref(false)

    const toggleReadReceipts = () => {
        readReceipts.value = !readReceipts.value
        store.read_receipts_enabled = true
    }

    const toggleTyping = () => {
        typingIndicators.value = !typingIndicators.value
        store.typing_indicators_enabled = true
    }

    const goToNotifications = async () => {
        store.currentPageId = 'SETTINGS_NOTIFICATIONS'
        await router.push({ name: 'SETTINGS_NOTIFICATIONS' })
    }

    const goHome = async () => {
        store.currentPageId = 'HOME'
        await router.push({ name: 'HOME' })
    }

    return {
        readReceipts,
        typingIndicators,
        toggleReadReceipts,
        toggleTyping,
        goToNotifications,
        goHome
    }
  }
}
</script>