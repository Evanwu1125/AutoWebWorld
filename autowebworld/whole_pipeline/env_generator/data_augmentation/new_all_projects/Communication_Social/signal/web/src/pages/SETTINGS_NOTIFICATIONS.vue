<template>
  <div class="h-screen bg-slate-50 flex flex-col">
    <header class="bg-white shadow-sm z-20 px-4 py-3 flex items-center justify-between">
        <div class="flex items-center">
            <button id="settings-notifications-back-privacy" @click="goBackPrivacy" class="p-2 text-slate-500 hover:text-blue-600 mr-4">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
                </svg>
            </button>
            <h1 class="text-xl font-bold text-slate-800">Notifications</h1>
        </div>
        <button 
            id="settings-notifications-save" 
            @click="save"
            :disabled="!sound"
            class="text-blue-600 font-semibold disabled:opacity-50 disabled:cursor-not-allowed hover:text-blue-700 transition-colors"
        >
            Save
        </button>
    </header>

    <div class="flex-1 overflow-y-auto p-4">
        <div class="max-w-md mx-auto space-y-6">
            <div class="bg-white rounded-2xl shadow-sm overflow-hidden divide-y divide-slate-100">
                <!-- Sound -->
                <div class="p-4 space-y-2">
                    <label class="font-medium text-slate-800 block">Notification Sound</label>
                    <div class="relative">
                        <button 
                            id="notification-sound-dropdown" 
                            @click="showSound = !showSound"
                            class="w-full bg-slate-100 rounded-xl px-4 py-3 flex items-center justify-between text-slate-700 font-medium"
                        >
                            <span class="capitalize">{{ sound || 'Select Sound' }}</span>
                            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
                            </svg>
                        </button>
                        
                        <div v-if="showSound" class="absolute top-full left-0 right-0 mt-2 bg-white border border-slate-100 rounded-xl shadow-lg z-10 overflow-hidden">
                            <div id="sound-option-chime" @click="selectSound('chime')" class="px-4 py-3 hover:bg-slate-50 cursor-pointer">Chime</div>
                            <div id="sound-option-ding" @click="selectSound('ding')" class="px-4 py-3 hover:bg-slate-50 cursor-pointer">Ding</div>
                            <div id="sound-option-silent" @click="selectSound('silent')" class="px-4 py-3 hover:bg-slate-50 cursor-pointer">Silent</div>
                        </div>
                    </div>
                </div>

                <!-- Vibrate -->
                <div class="p-4 flex items-center justify-between">
                    <div>
                        <h3 class="font-medium text-slate-800">Vibrate</h3>
                        <p class="text-sm text-slate-500">Vibrate on incoming messages</p>
                    </div>
                    <button 
                        id="notification-vibrate-toggle" 
                        @click="toggleVibrate"
                        :class="['w-12 h-6 rounded-full p-1 transition-colors', vibrate ? 'bg-blue-600' : 'bg-slate-300']"
                    >
                        <div :class="['bg-white w-4 h-4 rounded-full shadow-md transform transition-transform', vibrate ? 'translate-x-6' : 'translate-x-0']"></div>
                    </button>
                </div>
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
  name: 'SETTINGS_NOTIFICATIONS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const sound = ref(null)
    const vibrate = ref(false)
    const showSound = ref(false)

    const selectSound = (val) => {
        sound.value = val
        store.notification_sound = val
        showSound.value = false
    }

    const toggleVibrate = () => {
        vibrate.value = !vibrate.value
        store.notification_vibrate = true
    }

    const goBackPrivacy = async () => {
        store.currentPageId = 'SETTINGS_PRIVACY'
        await router.push({ name: 'SETTINGS_PRIVACY' })
    }

    const save = async () => {
        if (sound.value) {
            store.currentPageId = 'UPDATE_SETTINGS_SUCCESS'
            await router.push({ name: 'UPDATE_SETTINGS_SUCCESS' })
        }
    }

    return {
        sound,
        vibrate,
        showSound,
        selectSound,
        toggleVibrate,
        goBackPrivacy,
        save
    }
  }
}
</script>