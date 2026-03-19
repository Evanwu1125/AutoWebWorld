<template>
  <div class="min-h-screen bg-[#FAFBFC] flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
    <div class="max-w-md w-full space-y-8 bg-white p-8 rounded-lg shadow-md border border-gray-200">
      <div>
        <h2 class="mt-6 text-center text-3xl font-extrabold text-[#172B4D]">Create a new repository</h2>
        <p class="mt-2 text-center text-sm text-gray-600">
          A repository contains all project files, including the revision history.
        </p>
      </div>
      
      <div class="mt-8 space-y-6">
        <!-- Project/Workspace (Static for demo) -->
        <div>
           <label class="block text-sm font-medium text-gray-700">Project</label>
           <div class="mt-1 flex items-center p-2 border border-gray-300 rounded-md bg-gray-50 text-gray-500">
             <span class="mr-2">📁</span> Default Project
           </div>
        </div>

        <!-- Repository Name -->
        <div>
          <label for="repo-name-input" class="block text-sm font-medium text-gray-700">Repository Name <span class="text-red-500">*</span></label>
          <div class="mt-1">
            <input 
              id="repo-name-input" 
              name="repo-name" 
              type="text" 
              required 
              v-model="repoName"
              class="appearance-none block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
              placeholder="e.g. awesome-project"
            >
          </div>
        </div>

        <!-- Access Level Dropdown -->
        <div class="relative">
          <label class="block text-sm font-medium text-gray-700 mb-1">Access Level <span class="text-red-500">*</span></label>
          <button 
            id="repo-access-dropdown"
            @click="toggleDropdown"
            class="w-full bg-white border border-gray-300 rounded-md shadow-sm px-4 py-2 text-left cursor-default focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 sm:text-sm flex justify-between items-center"
          >
            <span class="block truncate">{{ accessLevel ? (accessLevel === 'private' ? 'Private' : 'Public') : 'Select access level' }}</span>
            <svg class="h-5 w-5 text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
              <path fill-rule="evenodd" d="M10 3a1 1 0 01.707.293l3 3a1 1 0 01-1.414 1.414L10 5.414 7.707 7.707a1 1 0 01-1.414-1.414l3-3A1 1 0 0110 3zm-3.707 9.293a1 1 0 011.414 0L10 14.586l2.293-2.293a1 1 0 011.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clip-rule="evenodd" />
            </svg>
          </button>

          <div v-if="isOpen" class="absolute mt-1 w-full bg-white shadow-lg max-h-60 rounded-md py-1 text-base ring-1 ring-black ring-opacity-5 overflow-auto focus:outline-none sm:text-sm z-10">
            <div 
              id="repo-access-private"
              @click="selectAccess('private')"
              class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-blue-50 group"
            >
              <div class="flex items-center">
                 <svg class="w-4 h-4 mr-2 text-gray-500 group-hover:text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"/></svg>
                 <span class="block truncate font-normal group-hover:font-semibold">Private</span>
              </div>
            </div>
            <div 
              id="repo-access-public"
              @click="selectAccess('public')"
              class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-blue-50 group"
            >
              <div class="flex items-center">
                 <svg class="w-4 h-4 mr-2 text-gray-500 group-hover:text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
                 <span class="block truncate font-normal group-hover:font-semibold">Public</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Description -->
        <div>
          <label for="repo-description-input" class="block text-sm font-medium text-gray-700">Description</label>
          <div class="mt-1">
            <textarea 
              id="repo-description-input" 
              name="repo-description" 
              rows="3" 
              v-model="repoDescription"
              class="shadow-sm focus:ring-blue-500 focus:border-blue-500 block w-full sm:text-sm border border-gray-300 rounded-md p-2"
              placeholder="What is this repository for?"
            ></textarea>
          </div>
        </div>

        <!-- Actions -->
        <div class="flex items-center justify-between pt-4">
          <button 
            id="create-repo-back" 
            @click="goBack" 
            type="button" 
            class="text-sm font-medium text-gray-600 hover:text-gray-900"
          >
            Cancel
          </button>
          <button 
            id="create-repo-submit" 
            @click="submitForm"
            type="submit" 
            class="group relative w-32 flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-[#0052CC] hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
            :disabled="!isValid"
          >
            Create
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
  name: 'CREATE_REPO_FORM',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const repoName = ref('')
    const repoDescription = ref('')
    const accessLevel = ref(null)
    const isOpen = ref(false)

    const toggleDropdown = () => {
      isOpen.value = !isOpen.value
    }

    const selectAccess = (level) => {
      accessLevel.value = level
      signatureStore.repo_access_level = level
      isOpen.value = false
    }

    // Two-way binding helpers for store updates via watch or directly updating store on submit
    // FSM says ACT_TYPE updates store immediately
    // So I should use v-model with computed setter or watch
    // Simple way: Update store in submit, OR watch refs.
    // FSM: ACT_CREATE_REPO_TYPE_NAME -> effects: set repo_name
    
    // Let's watch for strict FSM compliance
    // Actually, Vue v-model is easier. I'll just sync to store on input
    
    // Wait, the "Action Handlers" section says "Directly update Pinia store".
    // So I can just bind v-model="signatureStore.repo_name"? 
    // No, signatureStore ref might be null initially.
    
    // I will use local refs and sync to store, valid approach.
    // But FSM actions are fine-grained (type, click). 
    // I'll update store on every input event if I want to be super strict, or just use v-model to store ref directly if not null.
    // Let's use local refs and update store on change/input to avoid null issues if any.
    
    const isValid = computed(() => {
      return repoName.value.length > 0 && accessLevel.value !== null
    })

    // Sync to store for FSM state correctness
    const updateStore = () => {
       signatureStore.repo_name = repoName.value
       signatureStore.repo_description = repoDescription.value
       // accessLevel already updated store in selectAccess
    }

    const submitForm = async () => {
      updateStore()
      await router.push({ name: 'CREATE_REPO_REVIEW' })
    }

    const goBack = async () => {
      await router.push({ name: 'REPO_LIST' })
    }

    return {
      repoName,
      repoDescription,
      accessLevel,
      isOpen,
      isValid,
      toggleDropdown,
      selectAccess,
      submitForm,
      goBack
    }
  }
}
</script>