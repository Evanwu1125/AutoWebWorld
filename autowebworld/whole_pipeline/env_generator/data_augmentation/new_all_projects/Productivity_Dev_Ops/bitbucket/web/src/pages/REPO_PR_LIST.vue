<template>
  <div class="min-h-screen bg-[#FAFBFC] flex flex-col">
    <!-- Header -->
    <header class="bg-white border-b border-gray-200 px-6 py-4 flex items-center sticky top-0 z-20">
      <button 
        id="repo-pr-list-back" 
        @click="goBack" 
        class="mr-4 text-gray-500 hover:text-blue-600 transition-colors p-1 rounded-full hover:bg-gray-100"
      >
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"/></svg>
      </button>
      <h1 class="text-2xl font-bold text-[#172B4D]">Repository Pull Requests</h1>
    </header>

    <div class="flex-1 container mx-auto px-6 py-8">
      <!-- Search -->
      <div class="mb-6 max-w-lg">
        <div class="relative">
          <input 
            id="repo-pr-search-input"
            v-model="searchQuery"
            @keyup.enter="handleSearch"
            type="text" 
            placeholder="Search pull requests in this repository..."
            class="w-full pl-9 pr-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
          <svg class="w-4 h-4 text-gray-400 absolute left-3 top-2.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
        </div>
      </div>

      <!-- PR List -->
      <div id="repo-pr-list-container" class="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden min-h-[500px]">
        <div id="repo-pr-list" class="divide-y divide-gray-100">
          <div 
            v-for="pr in filteredPRs" 
            :key="pr.id"
            class="group p-4 flex items-start space-x-4 hover:bg-blue-50 transition-colors cursor-pointer"
            :class="{
              'pr-row-matched': hasSearched && matchesSearch(pr),
              'pr-row-visible': !hasSearched
            }"
            @click="openPR(pr)"
          >
            <!-- Avatar -->
            <div class="flex-shrink-0 w-10 h-10 rounded-full overflow-hidden bg-gray-100 border border-gray-200">
              <img :src="pr.image" alt="pr avatar" class="w-full h-full object-cover">
            </div>
            
            <div class="flex-1 min-w-0">
              <div class="flex items-center justify-between mb-1">
                <h3 class="text-base font-semibold text-gray-900 truncate group-hover:text-blue-600" :class="`data-id-${pr.id}`">
                  {{ pr.title }}
                </h3>
                <span 
                  class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium uppercase"
                  :class="{
                    'bg-green-100 text-green-800': pr.status === 'open',
                    'bg-purple-100 text-purple-800': pr.status === 'merged',
                    'bg-red-100 text-red-800': pr.status === 'declined'
                  }"
                >
                  {{ pr.status }}
                </span>
              </div>
              <p class="text-sm text-gray-600 mb-2">
                <span class="font-medium text-gray-800">#{{ pr.id.split('_')[1] }}</span> created by {{ pr.author_id }}
              </p>
              <div class="flex items-center text-xs text-gray-500">
                <span>Updated {{ pr.updated_at }}</span>
              </div>
            </div>
          </div>
          
          <!-- Empty State -->
          <div v-if="filteredPRs.length === 0" class="p-12 text-center text-gray-500">
             <img src="/images/PullRequests.jpg" alt="No PRs found" class="w-32 h-32 mx-auto mb-4 opacity-50">
             <p class="text-lg font-medium">No pull requests found for this repository</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'REPO_PR_LIST',
  setup() {
    const router = useRouter()
    const route = useRoute()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const repoId = route.params.repo_id || signatureStore.selected_repo_id
    const searchQuery = ref('')

    const filteredPRs = computed(() => {
      let result = dataStore.pull_requests.filter(pr => pr.repo_id === repoId)

      if (searchQuery.value) {
        const q = searchQuery.value.toLowerCase()
        result = result.filter(pr => pr.title.toLowerCase().includes(q))
      }
      return result
    })

    const hasSearched = computed(() => searchQuery.value.length > 0)
    
    const matchesSearch = (pr) => {
      if (!searchQuery.value) return false
      return pr.title.toLowerCase().includes(searchQuery.value.toLowerCase())
    }

    const handleSearch = () => {
      signatureStore.repo_pr_list_has_searched = true
      signatureStore.matched_repo_pr_id = filteredPRs.value.length > 0 ? filteredPRs.value[0].id : null
    }

    const openPR = async (pr) => {
      signatureStore.selected_pr_id = pr.id
      
      if (hasSearched.value) {
        signatureStore.repo_pr_list_has_searched = true
        signatureStore.matched_repo_pr_id = pr.id
      } else {
        signatureStore.repo_pr_list_viewport_anchor_id = pr.id
      }

      await router.push({ name: 'PR_DETAIL', params: { pr_id: pr.id } })
    }

    const goBack = async () => {
      signatureStore.currentPageId = 'REPO_DETAIL'
      await router.push({ name: 'REPO_DETAIL', params: { repo_id: repoId } })
    }

    return {
      searchQuery,
      filteredPRs,
      hasSearched,
      matchesSearch,
      handleSearch,
      openPR,
      goBack
    }
  }
}
</script>