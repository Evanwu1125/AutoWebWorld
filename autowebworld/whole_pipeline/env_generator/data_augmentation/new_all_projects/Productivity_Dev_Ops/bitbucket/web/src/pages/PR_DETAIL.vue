<template>
  <div class="min-h-screen bg-[#FAFBFC] flex flex-col">
    <!-- Header -->
    <header class="bg-white border-b border-gray-200 px-6 py-4 flex items-center sticky top-0 z-20">
      <button 
        id="pr-detail-back" 
        @click="goBack" 
        class="mr-4 text-gray-500 hover:text-blue-600 transition-colors p-1 rounded-full hover:bg-gray-100"
      >
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"/></svg>
      </button>
      
      <div v-if="pr" class="flex items-center">
         <span class="text-gray-500 mr-2">#{{ pr.id.split('_')[1] }}</span>
         <h1 class="text-xl font-bold text-[#172B4D]">{{ pr.title }}</h1>
         <span 
            class="ml-3 px-2 py-0.5 rounded text-xs font-bold uppercase"
            :class="{
              'bg-green-100 text-green-800': pr.status === 'open',
              'bg-purple-100 text-purple-800': pr.status === 'merged',
              'bg-red-100 text-red-800': pr.status === 'declined'
            }"
          >
            {{ pr.status }}
          </span>
      </div>
    </header>

    <main class="flex-1 container mx-auto px-6 py-8" v-if="pr">
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <!-- Main Content -->
        <div class="lg:col-span-2 space-y-6">
          <!-- Overview Card -->
          <div class="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
            <div class="flex items-center space-x-3 mb-4">
              <div class="w-10 h-10 rounded-full bg-gray-200 overflow-hidden">
                <img :src="pr.image" alt="author" class="w-full h-full object-cover">
              </div>
              <div>
                <p class="text-gray-900 font-medium">{{ pr.author_id }} <span class="text-gray-500 font-normal">wants to merge into</span> main</p>
                <p class="text-xs text-gray-500">{{ pr.created_at }}</p>
              </div>
            </div>
            
            <div class="prose max-w-none text-gray-700 bg-gray-50 p-4 rounded-md border border-gray-100">
               <p>This pull request implements key features required for the upcoming release.</p>
               <p>Please review carefully.</p>
            </div>
          </div>

          <!-- File Changes (Decorative) -->
          <div class="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
            <div class="px-4 py-3 bg-gray-50 border-b border-gray-200 font-medium text-gray-700">
              Files changed (3)
            </div>
            <div class="divide-y divide-gray-100">
              <div class="p-4 hover:bg-gray-50">src/components/Button.vue <span class="text-green-600 text-xs">+12</span> <span class="text-red-600 text-xs">-4</span></div>
              <div class="p-4 hover:bg-gray-50">src/utils/helpers.js <span class="text-green-600 text-xs">+45</span></div>
              <div class="p-4 hover:bg-gray-50">package.json <span class="text-green-600 text-xs">+1</span> <span class="text-red-600 text-xs">-1</span></div>
            </div>
          </div>
        </div>

        <!-- Sidebar -->
        <div class="space-y-6">
          <!-- Actions -->
          <div class="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
             <h3 class="font-bold text-[#172B4D] mb-4">Actions</h3>
             
             <button 
               v-if="pr.status === 'open'"
               id="merge-pr-button" 
               @click="goToMerge"
               class="w-full py-2 px-4 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-md shadow-sm transition-colors mb-3"
             >
               Merge
             </button>
             
             <button class="w-full py-2 px-4 bg-white border border-gray-300 text-gray-700 hover:bg-gray-50 font-medium rounded-md transition-colors">
               Decline
             </button>
          </div>

          <!-- Reviewers -->
          <div class="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
            <h3 class="font-bold text-[#172B4D] mb-4">Reviewers</h3>
            <div class="flex -space-x-2 overflow-hidden">
              <img class="inline-block h-8 w-8 rounded-full ring-2 ring-white" src="/images/Reviewers.jpg" alt="User 2" />
              <img class="inline-block h-8 w-8 rounded-full ring-2 ring-white" src="/images/photo1765608124.jpg" alt="User 3" />
              <button class="h-8 w-8 rounded-full bg-gray-100 flex items-center justify-center text-gray-500 text-xs ring-2 ring-white hover:bg-gray-200">+</button>
            </div>
          </div>
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
  name: 'PR_DETAIL',
  setup() {
    const route = useRoute()
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const prId = route.params.pr_id || signatureStore.selected_pr_id
    const pr = computed(() => dataStore.pull_requests.find(p => p.id === prId))

    const goBack = async () => {
      signatureStore.currentPageId = 'PR_LIST'
      await router.push({ name: 'PR_LIST' })
    }

    const goToMerge = async () => {
      signatureStore.currentPageId = 'MERGE_PR_FORM'
      await router.push({ name: 'MERGE_PR_FORM', params: { pr_id: prId } })
    }

    return {
      pr,
      goBack,
      goToMerge
    }
  }
}
</script>