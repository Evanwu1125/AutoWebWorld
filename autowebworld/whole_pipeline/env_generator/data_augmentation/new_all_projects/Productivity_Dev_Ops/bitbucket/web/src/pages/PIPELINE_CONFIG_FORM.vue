<template>
  <div class="min-h-screen bg-[#FAFBFC] flex items-center justify-center py-12 px-4">
    <div class="max-w-lg w-full bg-white p-8 rounded-lg shadow-md border border-gray-200">
      <h2 class="text-2xl font-bold text-[#172B4D] mb-6">Configure Pipeline</h2>
      
      <div class="space-y-6">
        <!-- Name -->
        <div>
          <label for="pipeline-name-input" class="block text-sm font-medium text-gray-700 mb-1">Pipeline Name <span class="text-red-500">*</span></label>
          <input 
            id="pipeline-name-input" 
            v-model="name"
            type="text" 
            class="shadow-sm focus:ring-blue-500 focus:border-blue-500 block w-full sm:text-sm border border-gray-300 rounded-md p-2"
          >
        </div>

        <!-- Trigger -->
        <div class="relative">
          <label class="block text-sm font-medium text-gray-700 mb-1">Trigger <span class="text-red-500">*</span></label>
          <button 
            id="pipeline-trigger-dropdown"
            @click="toggleTrigger"
            class="w-full bg-white border border-gray-300 rounded-md shadow-sm px-4 py-2 text-left flex justify-between items-center"
          >
            <span>{{ triggerLabel }}</span>
            <svg class="h-5 w-5 text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" /></svg>
          </button>
          <div v-if="isTriggerOpen" class="absolute mt-1 w-full bg-white shadow-lg rounded-md py-1 z-10 border border-gray-100">
             <div id="trigger-option-on-push" @click="selectTrigger('on_push')" class="px-4 py-2 hover:bg-gray-100 cursor-pointer">On Push</div>
             <div id="trigger-option-manual" @click="selectTrigger('manual')" class="px-4 py-2 hover:bg-gray-100 cursor-pointer">Manual</div>
          </div>
        </div>

        <!-- Branch -->
        <div class="relative">
          <label class="block text-sm font-medium text-gray-700 mb-1">Branch <span class="text-red-500">*</span></label>
          <button 
            id="pipeline-branch-dropdown"
            @click="toggleBranch"
            class="w-full bg-white border border-gray-300 rounded-md shadow-sm px-4 py-2 text-left flex justify-between items-center"
          >
            <span>{{ branch || 'Select branch' }}</span>
            <svg class="h-5 w-5 text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" /></svg>
          </button>
          <div v-if="isBranchOpen" class="absolute mt-1 w-full bg-white shadow-lg rounded-md py-1 z-10 border border-gray-100">
             <div id="pipeline-branch-main" @click="selectBranch('main')" class="px-4 py-2 hover:bg-gray-100 cursor-pointer">main</div>
             <div id="pipeline-branch-develop" @click="selectBranch('develop')" class="px-4 py-2 hover:bg-gray-100 cursor-pointer">develop</div>
          </div>
        </div>

        <!-- Actions -->
        <div class="flex justify-end space-x-4 pt-4 border-t border-gray-200">
           <button 
             id="pipeline-config-back" 
             @click="goBack"
             class="px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none"
           >
             Cancel
           </button>
           <button 
             id="pipeline-config-submit" 
             @click="submit"
             :disabled="!isValid"
             class="px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed"
           >
             Next
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
  name: 'PIPELINE_CONFIG_FORM',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const name = ref('')
    const trigger = ref(null)
    const branch = ref(null)
    
    const isTriggerOpen = ref(false)
    const isBranchOpen = ref(false)

    const toggleTrigger = () => isTriggerOpen.value = !isTriggerOpen.value
    const toggleBranch = () => isBranchOpen.value = !isBranchOpen.value

    const selectTrigger = (val) => {
      trigger.value = val
      signatureStore.pipeline_trigger = val
      isTriggerOpen.value = false
    }

    const selectBranch = (val) => {
      branch.value = val
      signatureStore.pipeline_branch = val
      isBranchOpen.value = false
    }
    
    const triggerLabel = computed(() => {
       if (trigger.value === 'on_push') return 'On Push'
       if (trigger.value === 'manual') return 'Manual'
       return 'Select trigger'
    })

    const isValid = computed(() => name.value.length > 0 && trigger.value && branch.value)

    const submit = async () => {
      signatureStore.pipeline_name = name.value
      // others set in select
      signatureStore.currentPageId = 'PIPELINE_CONFIG_REVIEW'
      await router.push({ name: 'PIPELINE_CONFIG_REVIEW' })
    }

    const goBack = async () => {
      signatureStore.currentPageId = 'PIPELINE_LIST'
      await router.push({ name: 'PIPELINE_LIST' })
    }

    return {
      name,
      trigger,
      branch,
      isTriggerOpen,
      isBranchOpen,
      triggerLabel,
      toggleTrigger,
      toggleBranch,
      selectTrigger,
      selectBranch,
      isValid,
      submit,
      goBack
    }
  }
}
</script>