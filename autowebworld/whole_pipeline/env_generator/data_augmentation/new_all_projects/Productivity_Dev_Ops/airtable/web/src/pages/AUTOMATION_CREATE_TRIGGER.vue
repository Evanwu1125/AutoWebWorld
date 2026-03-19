<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center p-6">
    <div class="bg-white rounded-xl shadow-xl w-full max-w-2xl overflow-hidden flex flex-col h-[600px]">
      
      <!-- Progress Bar -->
      <div class="h-2 bg-gray-100 w-full flex">
         <div class="h-full bg-blue-600 w-1/2"></div>
         <div class="h-full bg-gray-200 w-1/2"></div>
      </div>

      <div class="p-8 flex-1 flex flex-col">
         <div class="flex justify-between items-center mb-8">
           <h1 class="text-2xl font-bold text-gray-900">Choose a trigger</h1>
           <button id="back-automations-dashboard" @click="goBack" class="text-gray-400 hover:text-gray-600">
             Cancel
           </button>
         </div>

         <div class="flex-1 space-y-6">
           <p class="text-gray-600">What should start this automation?</p>
           
           <div class="relative">
             <label class="block text-sm font-medium text-gray-700 mb-2">Trigger</label>
             <button 
               id="trigger-dropdown"
               @click="dropdownOpen = !dropdownOpen"
               class="w-full flex items-center justify-between px-4 py-3 border border-gray-300 rounded-lg bg-white hover:border-blue-400 transition-colors shadow-sm"
             >
               <span class="flex items-center gap-2">
                  <span v-if="selectedTrigger" class="text-xl">
                    <span v-if="selectedTrigger === 'when-record-created'">📝</span>
                    <span v-else-if="selectedTrigger === 'when-record-updated'">🔄</span>
                    <span v-else>⏰</span>
                  </span>
                  {{ selectedTriggerLabel || 'Select a trigger...' }}
               </span>
               <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                 <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
               </svg>
             </button>
             
             <div v-if="dropdownOpen" class="absolute top-full left-0 w-full mt-2 bg-white border border-gray-200 rounded-lg shadow-xl z-20 overflow-hidden">
                <div id="trigger-record-created" @click="selectTrigger('when-record-created')" class="p-4 hover:bg-blue-50 cursor-pointer border-b border-gray-100 flex items-center gap-3">
                   <div class="bg-blue-100 p-2 rounded text-xl">📝</div>
                   <div>
                      <div class="font-bold text-gray-900">When record created</div>
                      <div class="text-xs text-gray-500">Triggers when a new record is added to a table</div>
                   </div>
                </div>
                <div id="trigger-record-updated" @click="selectTrigger('when-record-updated')" class="p-4 hover:bg-blue-50 cursor-pointer border-b border-gray-100 flex items-center gap-3">
                   <div class="bg-green-100 p-2 rounded text-xl">🔄</div>
                   <div>
                      <div class="font-bold text-gray-900">When record updated</div>
                      <div class="text-xs text-gray-500">Triggers when a field value changes</div>
                   </div>
                </div>
                <div id="trigger-scheduled-time" @click="selectTrigger('at-scheduled-time')" class="p-4 hover:bg-blue-50 cursor-pointer flex items-center gap-3">
                   <div class="bg-purple-100 p-2 rounded text-xl">⏰</div>
                   <div>
                      <div class="font-bold text-gray-900">At scheduled time</div>
                      <div class="text-xs text-gray-500">Triggers at a specific time or interval</div>
                   </div>
                </div>
             </div>
           </div>
         </div>

         <div class="flex justify-end pt-6 border-t border-gray-100">
            <button 
               id="next-to-action"
               @click="goNext"
               class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-8 rounded-lg shadow-md transition-all flex items-center gap-2"
               :disabled="!selectedTrigger"
               :class="{'opacity-50 cursor-not-allowed': !selectedTrigger}"
            >
               Next Step
               <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                 <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3" />
               </svg>
            </button>
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
  name: 'AUTOMATION_CREATE_TRIGGER',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const selectedTrigger = ref('')
    const dropdownOpen = ref(false)

    const selectedTriggerLabel = computed(() => {
       if (selectedTrigger.value === 'when-record-created') return 'When record created'
       if (selectedTrigger.value === 'when-record-updated') return 'When record updated'
       if (selectedTrigger.value === 'at-scheduled-time') return 'At scheduled time'
       return ''
    })

    const selectTrigger = (val) => {
      selectedTrigger.value = val
      store.trigger_type = val
      dropdownOpen.value = false
    }

    const goBack = async () => {
      store.setCurrentPageId('AUTOMATIONS_DASHBOARD')
      await router.push({ name: 'AUTOMATIONS_DASHBOARD' })
    }

    const goNext = async () => {
      store.setCurrentPageId('AUTOMATION_CREATE_ACTION')
      await router.push({ name: 'AUTOMATION_CREATE_ACTION' })
    }

    return {
      selectedTrigger,
      selectedTriggerLabel,
      dropdownOpen,
      selectTrigger,
      goBack,
      goNext
    }
  }
}
</script>