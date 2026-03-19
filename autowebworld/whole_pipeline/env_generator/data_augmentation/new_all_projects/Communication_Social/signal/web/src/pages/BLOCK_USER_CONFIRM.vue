<template>
  <div class="h-screen bg-slate-50 flex flex-col items-center justify-center p-4">
    <div class="bg-white w-full max-w-md rounded-2xl shadow-xl overflow-hidden p-6 space-y-6">
        <div class="text-center">
            <div class="w-16 h-16 bg-red-100 text-red-500 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
            </div>
            <h2 class="text-2xl font-bold text-slate-800">Block User?</h2>
            <p class="text-slate-500 mt-2">They won't be able to message or call you.</p>
        </div>

        <div class="space-y-2">
            <label class="block text-sm font-medium text-slate-700">Reason (Optional)</label>
            <textarea 
                id="block-reason-input"
                v-model="reason"
                rows="3"
                placeholder="Why are you blocking this user?"
                class="w-full bg-slate-100 rounded-xl p-3 focus:outline-none focus:ring-2 focus:ring-red-500 resize-none"
                @input="updateReason"
            ></textarea>
        </div>

        <div class="flex space-x-3 pt-2">
            <button 
                id="block-back-contact" 
                @click="goBack"
                class="flex-1 py-3 px-4 border border-slate-300 text-slate-700 font-semibold rounded-xl hover:bg-slate-50 transition-colors"
            >
                Cancel
            </button>
            <button 
                id="block-confirm-button" 
                @click="confirmBlock"
                :disabled="!reason.trim()"
                class="flex-1 py-3 px-4 bg-red-600 text-white font-semibold rounded-xl hover:bg-red-700 shadow-md disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
                Block
            </button>
        </div>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'BLOCK_USER_CONFIRM',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const reason = ref('')

    const updateReason = () => {
        store.block_report_reason = reason.value
    }

    const goBack = async () => {
        store.currentPageId = 'CONTACT_DETAIL'
        await router.push({ name: 'CONTACT_DETAIL' })
    }

    const confirmBlock = async () => {
        store.currentPageId = 'BLOCK_USER_SUCCESS'
        await router.push({ name: 'BLOCK_USER_SUCCESS' })
    }

    return {
        reason,
        updateReason,
        goBack,
        confirmBlock
    }
  }
}
</script>