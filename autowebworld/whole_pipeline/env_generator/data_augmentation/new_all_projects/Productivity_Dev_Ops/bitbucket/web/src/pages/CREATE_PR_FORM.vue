<template>
  <div class="min-h-screen bg-[#FAFBFC] flex items-center justify-center py-12 px-4">
    <div class="max-w-2xl w-full bg-white p-8 rounded-lg shadow-md border border-gray-200">
      <div class="mb-8">
        <h2 class="text-2xl font-bold text-[#172B4D]">Create Pull Request</h2>
      </div>

      <div class="space-y-6">
        <!-- Branches -->
        <div class="grid grid-cols-2 gap-4">
           <!-- Source -->
           <div class="relative">
             <label class="block text-sm font-medium text-gray-700 mb-1">Source branch <span class="text-red-500">*</span></label>
             <button 
               id="pr-source-branch-dropdown"
               @click="toggleSource"
               class="w-full bg-white border border-gray-300 rounded-md shadow-sm px-4 py-2 text-left flex justify-between items-center"
             >
               <span>{{ sourceBranch || 'Select source' }}</span>
               <svg class="h-5 w-5 text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" /></svg>
             </button>
             <div v-if="isSourceOpen" class="absolute mt-1 w-full bg-white shadow-lg rounded-md py-1 z-10 border border-gray-100">
               <div id="branch-option-main" @click="selectSource('main')" class="px-4 py-2 hover:bg-gray-100 cursor-pointer">main</div>
               <div id="branch-option-develop" @click="selectSource('develop')" class="px-4 py-2 hover:bg-gray-100 cursor-pointer">develop</div>
               <div id="branch-option-feature" @click="selectSource('feature')" class="px-4 py-2 hover:bg-gray-100 cursor-pointer">feature</div>
             </div>
           </div>
           
           <!-- Target -->
           <div class="relative">
             <label class="block text-sm font-medium text-gray-700 mb-1">Target branch <span class="text-red-500">*</span></label>
             <button 
               id="pr-target-branch-dropdown"
               @click="toggleTarget"
               class="w-full bg-white border border-gray-300 rounded-md shadow-sm px-4 py-2 text-left flex justify-between items-center"
             >
               <span>{{ targetBranch || 'Select target' }}</span>
               <svg class="h-5 w-5 text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" /></svg>
             </button>
             <div v-if="isTargetOpen" class="absolute mt-1 w-full bg-white shadow-lg rounded-md py-1 z-10 border border-gray-100">
               <div id="target-branch-main" @click="selectTarget('main')" class="px-4 py-2 hover:bg-gray-100 cursor-pointer">main</div>
               <div id="target-branch-develop" @click="selectTarget('develop')" class="px-4 py-2 hover:bg-gray-100 cursor-pointer">develop</div>
             </div>
           </div>
        </div>

        <!-- Title -->
        <div>
          <label for="pr-title-input" class="block text-sm font-medium text-gray-700 mb-1">Title <span class="text-red-500">*</span></label>
          <input 
            id="pr-title-input" 
            v-model="title"
            type="text" 
            class="shadow-sm focus:ring-blue-500 focus:border-blue-500 block w-full sm:text-sm border border-gray-300 rounded-md p-2"
          >
        </div>

        <!-- Description -->
        <div>
          <label for="pr-description-input" class="block text-sm font-medium text-gray-700 mb-1">Description</label>
          <textarea 
            id="pr-description-input" 
            v-model="description"
            rows="4" 
            class="shadow-sm focus:ring-blue-500 focus:border-blue-500 block w-full sm:text-sm border border-gray-300 rounded-md p-2"
          ></textarea>
        </div>

        <!-- Actions -->
        <div class="flex justify-end space-x-4 pt-4 border-t border-gray-200">
           <button 
             id="create-pr-back" 
             @click="goBack"
             class="px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none"
           >
             Cancel
           </button>
           <button 
             id="create-pr-submit" 
             @click="submit"
             :disabled="!isValid"
             class="px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed"
           >
             Create pull request
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
  name: 'CREATE_PR_FORM',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const title = ref('')
    const description = ref('')
    const sourceBranch = ref(null)
    const targetBranch = ref(null)
    
    const isSourceOpen = ref(false)
    const isTargetOpen = ref(false)

    const toggleSource = () => isSourceOpen.value = !isSourceOpen.value
    const toggleTarget = () => isTargetOpen.value = !isTargetOpen.value

    const selectSource = (val) => {
      sourceBranch.value = val
      signatureStore.pr_source_branch = val
      isSourceOpen.value = false
    }

    const selectTarget = (val) => {
      targetBranch.value = val
      signatureStore.pr_target_branch = val
      isTargetOpen.value = false
    }

    const isValid = computed(() => {
      return title.value.length > 0 && sourceBranch.value && targetBranch.value
    })

    const submit = async () => {
      signatureStore.pr_title = title.value
      signatureStore.pr_description = description.value
      // branches updated in select
      signatureStore.currentPageId = 'CREATE_PR_REVIEW'
      await router.push({ name: 'CREATE_PR_REVIEW' })
    }

    const goBack = async () => {
      signatureStore.currentPageId = 'PR_LIST'
      await router.push({ name: 'PR_LIST' })
    }

    return {
      title,
      description,
      sourceBranch,
      targetBranch,
      isSourceOpen,
      isTargetOpen,
      toggleSource,
      toggleTarget,
      selectSource,
      selectTarget,
      isValid,
      submit,
      goBack
    }
  }
}
</script>