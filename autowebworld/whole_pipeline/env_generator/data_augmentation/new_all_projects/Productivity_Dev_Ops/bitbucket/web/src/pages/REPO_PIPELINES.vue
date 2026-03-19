<template>
  <div class="min-h-screen bg-[#FAFBFC] flex flex-col">
    <!-- Header -->
    <header class="bg-white border-b border-gray-200 px-6 py-4 flex items-center sticky top-0 z-20">
      <button 
        id="repo-pipelines-back" 
        @click="goBack" 
        class="mr-4 text-gray-500 hover:text-blue-600 transition-colors p-1 rounded-full hover:bg-gray-100"
      >
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"/></svg>
      </button>
      <h1 class="text-2xl font-bold text-[#172B4D]">Repository Pipelines</h1>
    </header>

    <div class="flex-1 container mx-auto px-6 py-8">
      <!-- Pipeline List -->
      <div id="repo-pipelines-list-container" class="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden min-h-[500px]">
        <div id="repo-pipelines-list" class="divide-y divide-gray-100">
          <div 
            v-for="pipe in filteredPipelines" 
            :key="pipe.id"
            class="group p-4 flex items-center space-x-4 hover:bg-blue-50 transition-colors cursor-pointer pipeline-row-visible"
            @click="openPipeline(pipe)"
          >
            <!-- Status Icon -->
            <div class="flex-shrink-0 w-8 h-8 flex items-center justify-center rounded-full"
              :class="{
                'bg-green-100 text-green-600': pipe.status === 'success',
                'bg-red-100 text-red-600': pipe.status === 'failed',
                'bg-blue-100 text-blue-600 animate-pulse': pipe.status === 'running'
              }"
            >
              <svg v-if="pipe.status === 'success'" class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>
              <svg v-else-if="pipe.status === 'failed'" class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
              <svg v-else class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/></svg>
            </div>

            <!-- Pipeline Info -->
            <div class="flex-1 min-w-0">
              <div class="flex items-center justify-between mb-1">
                <h3 class="text-base font-semibold text-gray-900 truncate group-hover:text-blue-600" :class="`data-id-${pipe.id}`">
                  {{ pipe.name }} <span class="text-gray-400 font-normal">#{{ pipe.id.split('_')[1] }}</span>
                </h3>
                <span class="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">{{ pipe.branch }}</span>
              </div>
              <div class="flex items-center text-sm text-gray-600">
                <div class="flex items-center mr-4">
                   <svg class="w-4 h-4 mr-1 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
                   {{ pipe.trigger }}
                </div>
                <div class="flex items-center">
                   <svg class="w-4 h-4 mr-1 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
                   {{ pipe.created_at }}
                </div>
              </div>
            </div>
            
            <!-- Image -->
            <div class="hidden sm:block w-16 h-12 rounded bg-gray-100 overflow-hidden border border-gray-200">
               <img :src="pipe.image" alt="artifact" class="w-full h-full object-cover opacity-80" />
            </div>
          </div>
          
          <!-- Empty State -->
          <div v-if="filteredPipelines.length === 0" class="p-12 text-center text-gray-500">
             <img src="/images/photo1765608942.jpg" alt="No pipelines found" class="w-32 h-32 mx-auto mb-4 opacity-50">
             <p class="text-lg font-medium">No pipelines found for this repository</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'REPO_PIPELINES',
  setup() {
    const router = useRouter()
    const route = useRoute()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const repoId = route.params.repo_id || signatureStore.selected_repo_id
    
    const filteredPipelines = computed(() => {
      return dataStore.pipelines.filter(p => p.repo_id === repoId)
    })

    const openPipeline = async (pipe) => {
      signatureStore.selected_pipeline_id = pipe.id
      signatureStore.repo_pipelines_viewport_anchor_id = pipe.id
      await router.push({ name: 'PIPELINE_DETAIL', params: { pipeline_id: pipe.id } })
    }

    const goBack = async () => {
      signatureStore.currentPageId = 'REPO_DETAIL'
      await router.push({ name: 'REPO_DETAIL', params: { repo_id: repoId } })
    }

    return {
      filteredPipelines,
      openPipeline,
      goBack
    }
  }
}
</script>