<template>
  <div class="min-h-screen bg-[#FAFBFC] flex items-center justify-center py-12 px-4">
    <div class="max-w-lg w-full bg-white p-8 rounded-lg shadow-md border border-gray-200">
      <h2 class="text-2xl font-bold text-[#172B4D] mb-6">Merge Pull Request</h2>
      
      <div class="space-y-6">
        <!-- Strategy -->
        <div class="relative">
          <label class="block text-sm font-medium text-gray-700 mb-1">Merge Strategy <span class="text-red-500">*</span></label>
          <button 
            id="merge-strategy-dropdown"
            @click="toggleStrategy"
            class="w-full bg-white border border-gray-300 rounded-md shadow-sm px-4 py-2 text-left flex justify-between items-center"
          >
            <span>{{ strategyLabel }}</span>
            <svg class="h-5 w-5 text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" /></svg>
          </button>
          
          <div v-if="isStrategyOpen" class="absolute mt-1 w-full bg-white shadow-lg rounded-md py-1 z-10 border border-gray-100">
             <div id="merge-strategy-merge" @click="selectStrategy('merge_commit')" class="px-4 py-2 hover:bg-gray-100 cursor-pointer">
               <span class="font-medium block">Merge Commit</span>
               <span class="text-xs text-gray-500">Preserves all commits</span>
             </div>
             <div id="merge-strategy-squash" @click="selectStrategy('squash')" class="px-4 py-2 hover:bg-gray-100 cursor-pointer">
               <span class="font-medium block">Squash</span>
               <span class="text-xs text-gray-500">Combines into one commit</span>
             </div>
             <div id="merge-strategy-ff" @click="selectStrategy('fast_forward')" class="px-4 py-2 hover:bg-gray-100 cursor-pointer">
               <span class="font-medium block">Fast Forward</span>
               <span class="text-xs text-gray-500">No merge commit if possible</span>
             </div>
          </div>
        </div>

        <!-- Commit Message -->
        <div>
          <label for="merge-commit-message-input" class="block text-sm font-medium text-gray-700 mb-1">Commit Message <span class="text-red-500">*</span></label>
          <textarea 
            id="merge-commit-message-input" 
            v-model="commitMessage"
            rows="3" 
            class="shadow-sm focus:ring-blue-500 focus:border-blue-500 block w-full sm:text-sm border border-gray-300 rounded-md p-2"
          ></textarea>
        </div>

        <!-- Actions -->
        <div class="flex justify-end space-x-4 pt-4 border-t border-gray-200">
           <button 
             id="merge-pr-back" 
             @click="goBack"
             class="px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none"
           >
             Cancel
           </button>
           <button 
             id="merge-pr-submit" 
             @click="submit"
             :disabled="!isValid"
             class="px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed"
           >
             Merge
           </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'MERGE_PR_FORM',
  setup() {
    const router = useRouter()
    const route = useRoute()
    const signatureStore = useSignatureStore()

    const strategy = ref(null)
    const commitMessage = ref('')
    const isStrategyOpen = ref(false)

    const toggleStrategy = () => isStrategyOpen.value = !isStrategyOpen.value

    const selectStrategy = (val) => {
      strategy.value = val
      signatureStore.merge_strategy = val
      isStrategyOpen.value = false
    }

    const strategyLabel = computed(() => {
      if (strategy.value === 'merge_commit') return 'Merge Commit'
      if (strategy.value === 'squash') return 'Squash'
      if (strategy.value === 'fast_forward') return 'Fast Forward'
      return 'Select Strategy'
    })

    const isValid = computed(() => strategy.value && commitMessage.value.length > 0)

    const submit = async () => {
      signatureStore.merge_commit_message = commitMessage.value
      // strategy set in select
      signatureStore.currentPageId = 'MERGE_PR_REVIEW'
      await router.push({ name: 'MERGE_PR_REVIEW', params: route.params })
    }

    const goBack = async () => {
      signatureStore.currentPageId = 'PR_DETAIL'
      await router.push({ name: 'PR_DETAIL', params: route.params })
    }

    return {
      strategy,
      commitMessage,
      isStrategyOpen,
      strategyLabel,
      toggleStrategy,
      selectStrategy,
      isValid,
      submit,
      goBack
    }
  }
}
</script>