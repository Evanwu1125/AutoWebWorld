<template>
  <div class="h-screen bg-slate-50 flex flex-col">
    <header class="bg-white shadow-sm z-20 px-4 py-3 flex items-center justify-between">
        <div class="flex items-center">
            <button id="dm-back-info" @click="goBackInfo" class="p-2 text-slate-500 hover:text-blue-600 mr-4">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
                </svg>
            </button>
            <h1 class="text-xl font-bold text-slate-800">Disappearing Msgs</h1>
        </div>
        <button 
            id="dm-save" 
            @click="save"
            :disabled="timerValue === 0"
            class="text-blue-600 font-semibold disabled:opacity-50 disabled:cursor-not-allowed hover:text-blue-700 transition-colors"
        >
            Save
        </button>
    </header>

    <div class="flex-1 p-6 flex flex-col items-center">
        <div class="w-full max-w-md bg-white p-6 rounded-2xl shadow-sm text-center">
            <div class="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
            </div>
            <h2 class="text-lg font-bold text-slate-900 mb-2">Set Timer</h2>
            <p class="text-slate-500 mb-8">Messages in this chat will disappear after the selected time.</p>

            <div class="mb-2 text-3xl font-bold text-blue-600">{{ timerLabel }}</div>
            
            <input 
                id="dm-slider"
                type="range" 
                min="0" 
                max="604800" 
                step="3600"
                v-model="timerValue"
                @input="updateTimer"
                class="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600 mb-4"
            />
            <div class="flex justify-between text-xs text-slate-400 font-medium px-1">
                <span>Off</span>
                <span>1 Week</span>
            </div>
        </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'DISAPPEARING_MESSAGES_SETTINGS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const timerValue = ref(0)

    const timerLabel = computed(() => {
        if (timerValue.value == 0) return 'Off'
        if (timerValue.value < 86400) return `${Math.floor(timerValue.value / 3600)} hours`
        return `${Math.floor(timerValue.value / 86400)} days`
    })

    const updateTimer = () => {
        store.disappearing_timer_seconds = parseInt(timerValue.value)
    }

    const goBackInfo = async () => {
        store.currentPageId = 'CHAT_INFO'
        await router.push({ name: 'CHAT_INFO' })
    }

    const save = async () => {
        if (timerValue.value > 0) {
            store.currentPageId = 'CHAT_THREAD'
            await router.push({ name: 'CHAT_THREAD' })
        }
    }

    return {
        timerValue,
        timerLabel,
        updateTimer,
        goBackInfo,
        save
    }
  }
}
</script>