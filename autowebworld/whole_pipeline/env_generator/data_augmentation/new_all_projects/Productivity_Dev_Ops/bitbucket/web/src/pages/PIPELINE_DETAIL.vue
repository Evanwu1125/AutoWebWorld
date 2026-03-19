<template>
  <div class="min-h-screen bg-[#FAFBFC] flex flex-col">
    <!-- Header -->
    <header class="bg-white border-b border-gray-200 px-6 py-4 flex items-center sticky top-0 z-20">
      <button 
        id="pipeline-detail-back" 
        @click="goBack" 
        class="mr-4 text-gray-500 hover:text-blue-600 transition-colors p-1 rounded-full hover:bg-gray-100"
      >
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"/></svg>
      </button>
      
      <div v-if="pipe" class="flex items-center">
         <span class="text-gray-500 mr-2">#{{ pipe.id.split('_')[1] }}</span>
         <h1 class="text-xl font-bold text-[#172B4D]">{{ pipe.name }}</h1>
         <span 
            class="ml-3 px-2 py-0.5 rounded text-xs font-bold uppercase"
            :class="{
              'bg-green-100 text-green-800': pipe.status === 'success',
              'bg-red-100 text-red-800': pipe.status === 'failed',
              'bg-blue-100 text-blue-800 animate-pulse': pipe.status === 'running'
            }"
          >
            {{ pipe.status }}
          </span>
      </div>
    </header>

    <main class="flex-1 container mx-auto px-6 py-8" v-if="pipe">
      <!-- Steps Visualization -->
      <div class="bg-white p-6 rounded-lg shadow-sm border border-gray-200 mb-6">
        <h3 class="font-bold text-[#172B4D] mb-6">Pipeline Steps</h3>
        <div class="flex items-center justify-between relative">
           <!-- Line -->
           <div class="absolute top-1/2 left-0 w-full h-1 bg-gray-200 -z-0"></div>
           
           <!-- Step 1 -->
           <div class="relative z-10 flex flex-col items-center bg-white px-4">
             <div class="w-10 h-10 rounded-full flex items-center justify-center text-white font-bold"
               :class="pipe.status === 'failed' ? 'bg-green-500' : 'bg-green-500'"
             >1</div>
             <span class="mt-2 text-sm font-medium">Build</span>
             <span class="text-xs text-gray-500">2m 30s</span>
           </div>

           <!-- Step 2 -->
           <div class="relative z-10 flex flex-col items-center bg-white px-4">
             <div class="w-10 h-10 rounded-full flex items-center justify-center text-white font-bold"
                :class="pipe.status === 'failed' ? 'bg-red-500' : (pipe.status === 'running' ? 'bg-blue-500 animate-pulse' : 'bg-green-500')"
             >2</div>
             <span class="mt-2 text-sm font-medium">Test</span>
             <span class="text-xs text-gray-500">4m 10s</span>
           </div>

           <!-- Step 3 -->
           <div class="relative z-10 flex flex-col items-center bg-white px-4">
             <div class="w-10 h-10 rounded-full flex items-center justify-center text-white font-bold border-2"
                :class="pipe.status === 'success' ? 'bg-green-500 border-green-500' : 'bg-white border-gray-300 text-gray-400'"
             >3</div>
             <span class="mt-2 text-sm font-medium">Deploy</span>
             <span class="text-xs text-gray-500">Pending</span>
           </div>
        </div>
      </div>

      <!-- Logs -->
      <div class="bg-[#172B4D] text-gray-300 p-6 rounded-lg shadow-sm border border-gray-800 font-mono text-sm overflow-x-auto">
        <div class="mb-2 text-green-400">$ git clone repository...</div>
        <div class="mb-2">Cloning into 'project'...</div>
        <div class="mb-2">Checking out main...</div>
        <div class="mb-2 text-green-400">$ npm install</div>
        <div class="mb-2">Added 1234 packages in 5s</div>
        <div class="mb-2 text-green-400">$ npm test</div>
        <div v-if="pipe.status === 'failed'">
           <div class="text-red-400">FAIL src/App.test.js</div>
           <div class="text-red-400">  ● renders learn react link</div>
           <div class="text-red-400">    expect(linkElement).toBeInTheDocument();</div>
        </div>
        <div v-else>
           <div class="text-green-400">PASS src/App.test.js</div>
           <div class="text-green-400">PASS src/components/Button.test.js</div>
        </div>
        <div class="mt-4 border-t border-gray-700 pt-2 animate-pulse" v-if="pipe.status === 'running'">_</div>
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
  name: 'PIPELINE_DETAIL',
  setup() {
    const route = useRoute()
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const pipeId = route.params.pipeline_id || signatureStore.selected_pipeline_id
    const pipe = computed(() => dataStore.pipelines.find(p => p.id === pipeId))

    const goBack = async () => {
      signatureStore.currentPageId = 'PIPELINE_LIST'
      await router.push({ name: 'PIPELINE_LIST' })
    }

    return {
      pipe,
      goBack
    }
  }
}
</script>