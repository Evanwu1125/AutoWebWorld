<template>
  <div class="min-h-screen bg-[#FAFBFC] flex flex-col">
    <!-- Header -->
    <header class="bg-white border-b border-gray-200 px-6 py-4 flex items-center sticky top-0 z-20">
      <button 
        id="repo-settings-back" 
        @click="goBack" 
        class="mr-4 text-gray-500 hover:text-blue-600 transition-colors p-1 rounded-full hover:bg-gray-100"
      >
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"/></svg>
      </button>
      <h1 class="text-2xl font-bold text-[#172B4D]">Repository Settings</h1>
    </header>

    <div class="flex-1 container mx-auto px-6 py-8 max-w-2xl">
      <div class="bg-white p-8 rounded-lg shadow-sm border border-gray-200">
         <h2 class="text-xl font-bold text-[#172B4D] mb-6">General</h2>
         
         <div class="space-y-8">
            <!-- Default Branch -->
            <div class="relative">
              <label class="block text-sm font-medium text-gray-700 mb-1">Default Branch</label>
              <p class="text-xs text-gray-500 mb-2">The branch that is selected by default when viewing the repository.</p>
              
              <button 
                id="default-branch-dropdown"
                @click="toggleBranch"
                class="w-full bg-white border border-gray-300 rounded-md shadow-sm px-4 py-2 text-left flex justify-between items-center"
              >
                <span>{{ branch || 'Select branch' }}</span>
                <svg class="h-5 w-5 text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" /></svg>
              </button>
              
              <div v-if="isBranchOpen" class="absolute mt-1 w-full bg-white shadow-lg rounded-md py-1 z-10 border border-gray-100">
                 <div id="default-branch-main" @click="selectBranch('main')" class="px-4 py-2 hover:bg-gray-100 cursor-pointer flex items-center">
                   <svg class="w-4 h-4 mr-2 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"/></svg>
                   main
                 </div>
                 <div id="default-branch-develop" @click="selectBranch('develop')" class="px-4 py-2 hover:bg-gray-100 cursor-pointer flex items-center">
                   <svg class="w-4 h-4 mr-2 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"/></svg>
                   develop
                 </div>
              </div>
            </div>

            <!-- Other Settings (Static/Decorative) -->
            <div>
               <label class="flex items-center space-x-2">
                 <input type="checkbox" class="rounded text-blue-600 focus:ring-blue-500" checked>
                 <span class="text-sm text-gray-700">Allow force pushes</span>
               </label>
            </div>
            <div>
               <label class="flex items-center space-x-2">
                 <input type="checkbox" class="rounded text-blue-600 focus:ring-blue-500" checked>
                 <span class="text-sm text-gray-700">Allow fork syncing</span>
               </label>
            </div>
         </div>
      </div>
      
      <div class="mt-8 bg-red-50 p-6 rounded-lg border border-red-200">
         <h3 class="text-red-800 font-bold mb-2">Danger Zone</h3>
         <p class="text-red-600 text-sm mb-4">Once you delete a repository, there is no going back. Please be certain.</p>
         <button class="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-md text-sm font-medium">Delete Repository</button>
      </div>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'REPO_SETTINGS',
  setup() {
    const router = useRouter()
    const route = useRoute()
    const signatureStore = useSignatureStore()

    const branch = ref('main')
    const isBranchOpen = ref(false)

    const toggleBranch = () => isBranchOpen.value = !isBranchOpen.value

    const selectBranch = (val) => {
      branch.value = val
      signatureStore.default_branch = val
      isBranchOpen.value = false
    }

    const goBack = async () => {
      signatureStore.currentPageId = 'REPO_DETAIL'
      await router.push({ name: 'REPO_DETAIL', params: route.params })
    }

    return {
      branch,
      isBranchOpen,
      toggleBranch,
      selectBranch,
      goBack
    }
  }
}
</script>