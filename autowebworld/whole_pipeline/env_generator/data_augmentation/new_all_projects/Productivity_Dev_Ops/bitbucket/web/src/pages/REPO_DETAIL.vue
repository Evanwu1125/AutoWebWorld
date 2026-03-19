<template>
  <div class="min-h-screen bg-[#FAFBFC] flex flex-col">
    <!-- Header with Back Button -->
    <header class="bg-white border-b border-gray-200 px-6 py-4 flex items-center sticky top-0 z-20">
      <button 
        id="repo-detail-back" 
        @click="goBack" 
        class="mr-4 text-gray-500 hover:text-blue-600 transition-colors p-1 rounded-full hover:bg-gray-100"
      >
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"/></svg>
      </button>
      
      <div v-if="repo" class="flex items-center">
         <div class="w-8 h-8 rounded overflow-hidden mr-3 bg-gray-200">
           <img :src="repo.image" alt="repo icon" class="w-full h-full object-cover" />
         </div>
         <div>
            <h1 class="text-xl font-bold text-[#172B4D]">
              <span class="font-normal text-gray-500">{{ repo.owner }} /</span> {{ repo.name }}
            </h1>
         </div>
      </div>
    </header>

    <main class="flex-1 container mx-auto px-6 py-8">
      <!-- Repo Navigation Cards -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        
        <!-- Pull Requests -->
        <div 
          id="repo-nav-pull-requests" 
          @click="goToPRList"
          class="bg-white p-6 rounded-lg shadow-sm border border-gray-200 hover:shadow-md hover:border-blue-300 cursor-pointer transition-all group"
        >
          <div class="flex items-center justify-between mb-4">
             <div class="p-3 bg-green-100 text-green-600 rounded-lg group-hover:bg-green-600 group-hover:text-white transition-colors">
               <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4"/></svg>
             </div>
             <span class="text-2xl font-bold text-gray-700">{{ openPRCount }}</span>
          </div>
          <h3 class="text-lg font-semibold text-[#172B4D] mb-1">Pull Requests</h3>
          <p class="text-sm text-gray-500">Review and merge code changes.</p>
        </div>

        <!-- Pipelines -->
        <div 
          id="repo-nav-pipelines" 
          @click="goToPipelines"
          class="bg-white p-6 rounded-lg shadow-sm border border-gray-200 hover:shadow-md hover:border-purple-300 cursor-pointer transition-all group"
        >
          <div class="flex items-center justify-between mb-4">
             <div class="p-3 bg-purple-100 text-purple-600 rounded-lg group-hover:bg-purple-600 group-hover:text-white transition-colors">
               <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
             </div>
             <div class="flex items-center text-sm font-medium text-green-600" v-if="lastPipelineSuccess">
                <span class="w-2 h-2 bg-green-500 rounded-full mr-2"></span> Passing
             </div>
          </div>
          <h3 class="text-lg font-semibold text-[#172B4D] mb-1">Pipelines</h3>
          <p class="text-sm text-gray-500">View CI/CD builds and deployments.</p>
        </div>

        <!-- Settings -->
        <div 
          id="repo-nav-settings" 
          @click="goToSettings"
          class="bg-white p-6 rounded-lg shadow-sm border border-gray-200 hover:shadow-md hover:border-gray-400 cursor-pointer transition-all group"
        >
          <div class="flex items-center justify-between mb-4">
             <div class="p-3 bg-gray-100 text-gray-600 rounded-lg group-hover:bg-gray-600 group-hover:text-white transition-colors">
               <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/></svg>
             </div>
          </div>
          <h3 class="text-lg font-semibold text-[#172B4D] mb-1">Settings</h3>
          <p class="text-sm text-gray-500">Configure repository options.</p>
        </div>
      </div>

      <!-- Repo Activity / Readme Preview (Decorative) -->
      <div class="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
        <h2 class="text-xl font-bold text-[#172B4D] mb-4">README.md</h2>
        <div class="prose max-w-none text-gray-600">
           <p>This is the main repository for the <strong>{{ repo ? repo.name : 'project' }}</strong>.</p>
           <p>It contains source code, documentation, and configuration files.</p>
           <hr class="my-4">
           <h4>Getting Started</h4>
           <pre class="bg-gray-800 text-gray-100 p-4 rounded-md overflow-x-auto">git clone https://bitbucket.org/{{ repo ? repo.owner.replace(/\s+/g, '').toLowerCase() : 'user' }}/{{ repo ? repo.name : 'repo' }}.git
cd {{ repo ? repo.name : 'repo' }}
npm install
npm start</pre>
        </div>
      </div>

    </main>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'REPO_DETAIL',
  setup() {
    const route = useRoute()
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const repoId = route.params.repo_id || signatureStore.selected_repo_id
    
    const repo = computed(() => dataStore.repositories.find(r => r.id === repoId))
    
    // Stats
    const openPRCount = computed(() => 
      dataStore.pull_requests.filter(pr => pr.repo_id === repoId && pr.status === 'open').length
    )
    
    const lastPipelineSuccess = computed(() => {
       const pipes = dataStore.pipelines.filter(p => p.repo_id === repoId)
       if (pipes.length === 0) return true // Default
       return pipes[0].status === 'success'
    })

    const goBack = async () => {
      signatureStore.currentPageId = 'REPO_LIST'
      await router.push({ name: 'REPO_LIST' })
    }

    const goToPRList = async () => {
      signatureStore.currentPageId = 'REPO_PR_LIST'
      // signatureStore.selected_repo_id already set
      await router.push({ name: 'REPO_PR_LIST', params: { repo_id: repoId } })
    }

    const goToPipelines = async () => {
      signatureStore.currentPageId = 'REPO_PIPELINES'
      await router.push({ name: 'REPO_PIPELINES', params: { repo_id: repoId } })
    }

    const goToSettings = async () => {
      signatureStore.currentPageId = 'REPO_SETTINGS'
      await router.push({ name: 'REPO_SETTINGS', params: { repo_id: repoId } })
    }

    return {
      repo,
      openPRCount,
      lastPipelineSuccess,
      goBack,
      goToPRList,
      goToPipelines,
      goToSettings
    }
  }
}
</script>