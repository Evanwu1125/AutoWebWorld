<template>
  <div class="h-screen bg-slate-50 flex flex-col items-center justify-center p-4">
    <div class="bg-white w-full max-w-md rounded-2xl shadow-xl overflow-hidden p-6 space-y-6">
        <h2 class="text-2xl font-bold text-slate-800 text-center">Confirm Message</h2>
        
        <div class="bg-slate-50 p-4 rounded-xl border border-slate-100">
            <p class="text-slate-600 font-medium">Message Preview:</p>
            <p class="text-slate-800 mt-2 italic">"{{ draftText }}"</p>
        </div>

        <div class="space-y-4">
            <!-- Read Receipts Toggle -->
            <div class="flex items-center justify-between">
                <span class="text-slate-700 font-medium">Read Receipts</span>
                <button 
                    id="read-receipt-toggle" 
                    @click="toggleReadReceipts"
                    :class="['w-12 h-6 rounded-full p-1 transition-colors', readReceipts ? 'bg-blue-600' : 'bg-slate-300']"
                >
                    <div :class="['bg-white w-4 h-4 rounded-full shadow-md transform transition-transform', readReceipts ? 'translate-x-6' : 'translate-x-0']"></div>
                </button>
            </div>

            <!-- Disappearing Timer Slider -->
            <div class="space-y-2">
                <div class="flex justify-between">
                    <span class="text-slate-700 font-medium">Disappearing Timer</span>
                    <span class="text-blue-600 text-sm font-bold">{{ timerLabel }}</span>
                </div>
                <input 
                    id="disappearing-slider"
                    type="range" 
                    min="0" 
                    max="604800" 
                    step="3600"
                    v-model="timerValue"
                    @input="updateTimer"
                    class="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                />
            </div>
        </div>

        <div class="flex space-x-3 pt-4">
            <button 
                id="back-thread" 
                @click="goBack"
                class="flex-1 py-3 px-4 border border-slate-300 text-slate-700 font-semibold rounded-xl hover:bg-slate-50 transition-colors"
            >
                Cancel
            </button>
            <button 
                id="confirm-send" 
                @click="confirmSend"
                class="flex-1 py-3 px-4 bg-blue-600 text-white font-semibold rounded-xl hover:bg-blue-700 shadow-md transition-colors"
            >
                Send Now
            </button>
        </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'SEND_MESSAGE_CONFIRM',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const draftText = computed(() => store.draft_message_text)
    const readReceipts = ref(false)
    const timerValue = ref(0)

    const timerLabel = computed(() => {
        if (timerValue.value === 0) return 'Off'
        if (timerValue.value < 3600) return `${timerValue.value / 60} min`
        if (timerValue.value < 86400) return `${timerValue.value / 3600} hr`
        return `${timerValue.value / 86400} days`
    })

    const toggleReadReceipts = () => {
        readReceipts.value = !readReceipts.value
        store.send_read_receipts = true // Effect only sets to true in FSM, but toggling is implied by UI logic
    }

    const updateTimer = () => {
        store.send_disappearing_timer_seconds = parseInt(timerValue.value)
    }

    const goBack = async () => {
        store.currentPageId = 'CHAT_THREAD'
        await router.push({ name: 'CHAT_THREAD' })
    }

    const confirmSend = async () => {
        store.currentPageId = 'SEND_MESSAGE_SUCCESS'
        await router.push({ name: 'SEND_MESSAGE_SUCCESS' })
    }

    return {
        draftText,
        readReceipts,
        timerValue,
        timerLabel,
        toggleReadReceipts,
        updateTimer,
        goBack,
        confirmSend
    }
  }
}
</script>