<template>
  <div class="h-screen bg-slate-50 flex flex-col items-center justify-center p-4">
    <div class="bg-white w-full max-w-sm rounded-2xl shadow-xl p-6 space-y-6 text-center">
        <h2 class="text-2xl font-bold text-slate-800">Start Call</h2>

        <div class="flex flex-col items-center">
            <img :src="contact.avatar" class="w-24 h-24 rounded-full object-cover mb-2 border-4 border-slate-50" />
            <h3 class="text-xl font-semibold text-slate-900">{{ contact.name }}</h3>
        </div>

        <div class="space-y-2 text-left">
             <label class="text-sm font-medium text-slate-700 block">Call Type</label>
             <div class="relative">
                 <button 
                    id="call-type-dropdown" 
                    @click="showType = !showType"
                    class="w-full bg-slate-100 rounded-xl px-4 py-3 flex items-center justify-between text-slate-700 font-medium"
                 >
                    <span>{{ selectedType ? (selectedType === 'voice' ? 'Voice Call' : 'Video Call') : 'Select Type' }}</span>
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
                    </svg>
                 </button>

                 <div v-if="showType" class="absolute top-full left-0 right-0 mt-2 bg-white border border-slate-100 rounded-xl shadow-lg z-10 overflow-hidden">
                     <div 
                        id="call-type-option-voice" 
                        @click="selectType('voice')"
                        class="px-4 py-3 hover:bg-slate-50 cursor-pointer flex items-center space-x-2"
                     >
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
                        </svg>
                        <span>Voice Call</span>
                     </div>
                     <div 
                        id="call-type-option-video" 
                        @click="selectType('video')"
                        class="px-4 py-3 hover:bg-slate-50 cursor-pointer flex items-center space-x-2"
                     >
                         <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                         </svg>
                        <span>Video Call</span>
                     </div>
                 </div>
             </div>
        </div>

        <div class="space-y-3 pt-2">
            <button 
                id="start-call-button" 
                @click="startCall"
                :disabled="!selectedType"
                class="w-full py-3 px-4 bg-green-500 text-white font-semibold rounded-xl hover:bg-green-600 shadow-md disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
                Start Call
            </button>
            <button 
                id="start-call-back-history" 
                @click="goBack"
                class="w-full py-3 px-4 bg-slate-100 text-slate-700 font-semibold rounded-xl hover:bg-slate-200 transition-colors"
            >
                Cancel
            </button>
        </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'START_CALL_SETUP',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const showType = ref(false)
    const selectedType = ref(null)

    // selected_call_id points to a call history item
    // OR we might be starting a call from scratch (but FSM flow is via history or other means)
    // Actually FSM says: CALLS_OPEN_FILTERED_CALL -> START_CALL_SETUP
    // So selected_call_id is set.
    const callId = computed(() => store.selected_call_id)
    const call = computed(() => dataStore.calls.find(c => c.id === callId.value))
    const contact = computed(() => {
        if (!call.value) return { name: 'Unknown', avatar: '/images/photo1765611489.jpg' }
        return dataStore.contacts.find(c => c.id === call.value.contact_id) || { name: 'Unknown', avatar: '/images/photo1765611489.jpg' }
    })

    const selectType = (type) => {
        selectedType.value = type
        store.call_type = type
        showType.value = false
    }

    const startCall = async () => {
        store.currentPageId = 'START_CALL_SUCCESS'
        await router.push({ name: 'START_CALL_SUCCESS' })
    }

    const goBack = async () => {
        store.currentPageId = 'CALL_HISTORY'
        await router.push({ name: 'CALL_HISTORY' })
    }

    return {
        contact,
        showType,
        selectedType,
        selectType,
        startCall,
        goBack
    }
  }
}
</script>