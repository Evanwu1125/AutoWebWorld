<template>
  <div class="min-h-screen bg-slate-50 py-12">
    <div class="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="bg-white shadow rounded-lg overflow-hidden">
        <div class="px-4 py-5 sm:p-6 space-y-8">
          <h2 class="text-xl font-bold text-slate-900">What triggers this flow?</h2>
          
          <div class="relative">
            <button 
              id="trigger-type-dropdown"
              @click="toggleDropdown"
              class="w-full bg-white border border-slate-300 rounded-lg py-4 px-4 flex items-center justify-between shadow-sm hover:border-blue-500 focus:outline-none transition-colors"
            >
              <span class="block font-medium text-slate-700">
                {{ selectedLabel || 'Select a trigger...' }}
              </span>
              <span class="pointer-events-none flex items-center">
                <svg class="h-5 w-5 text-slate-400" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 3a1 1 0 01.707.293l3 3a1 1 0 01-1.414 1.414L10 5.414 7.707 7.707a1 1 0 01-1.414-1.414l3-3A1 1 0 0110 3zm-3.707 9.293a1 1 0 011.414 0L10 14.586l2.293-2.293a1 1 0 011.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clip-rule="evenodd" /></svg>
              </span>
            </button>

            <div v-if="isOpen" class="absolute z-10 mt-1 w-full bg-white shadow-xl rounded-lg py-1 ring-1 ring-black ring-opacity-5">
              <div 
                id="trigger-type-event"
                @click="selectTrigger('event')"
                class="cursor-pointer select-none relative py-3 pl-4 pr-9 hover:bg-blue-50 hover:text-blue-900 border-b border-slate-50"
              >
                <div class="font-medium">Metric / Event</div>
                <div class="text-xs text-slate-500">Trigger when someone performs an action (e.g. Placed Order)</div>
              </div>

              <div 
                id="trigger-type-segment"
                @click="selectTrigger('segment')"
                class="cursor-pointer select-none relative py-3 pl-4 pr-9 hover:bg-blue-50 hover:text-blue-900"
              >
                <div class="font-medium">Segment Join</div>
                <div class="text-xs text-slate-500">Trigger when someone enters a segment</div>
              </div>
            </div>
          </div>
        </div>

        <div class="px-4 py-4 bg-slate-50 border-t border-slate-200 sm:px-6 flex justify-between">
          <button 
            id="back-flows-list"
            @click="goBack"
            class="inline-flex justify-center py-2 px-4 border border-slate-300 shadow-sm text-sm font-medium rounded-md text-slate-700 bg-white hover:bg-slate-50 focus:outline-none"
          >
            Cancel
          </button>
          <button 
            id="btn-trigger-continue"
            @click="goContinue"
            :disabled="!isValid"
            class="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none disabled:opacity-50"
          >
            Continue
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
  name: 'FLOW_TRIGGER_SETUP',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const isOpen = ref(false)

    const selectedLabel = computed(() => {
      if (store.flow_trigger_type === 'event') return 'Metric / Event'
      if (store.flow_trigger_type === 'segment') return 'Segment Join'
      return ''
    })

    function toggleDropdown() {
      isOpen.value = !isOpen.value
    }

    function selectTrigger(type) {
      store.flow_trigger_type = type
      isOpen.value = false
    }

    const isValid = computed(() => {
      return store.flow_trigger_type && store.flow_trigger_type.length > 0
    })

    async function goBack() {
      store.setCurrentPageId('FLOWS_LIST')
      await router.push({ name: 'FLOWS_LIST' })
    }

    async function goContinue() {
      if (!isValid.value) return
      store.setCurrentPageId('FLOW_EMAIL_CONTENT')
      await router.push({ name: 'FLOW_EMAIL_CONTENT' })
    }

    return {
      isOpen,
      selectedLabel,
      toggleDropdown,
      selectTrigger,
      isValid,
      goBack,
      goContinue
    }
  }
}
</script>