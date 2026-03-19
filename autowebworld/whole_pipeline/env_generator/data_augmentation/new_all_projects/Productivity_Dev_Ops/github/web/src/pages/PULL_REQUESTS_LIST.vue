<template>
  <div class="min-h-screen bg-[#0d1117] text-white font-sans">
    <!-- Header -->
    <header class="bg-[#161b22] border-b border-gray-700 py-4 px-6 flex items-center justify-between">
      <div class="flex items-center gap-4">
        <div id="pulls-back-to-repo" class="cursor-pointer flex items-center gap-1 text-blue-400 hover:underline" @click="goBack">
          <svg height="16" aria-hidden="true" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-arrow-left fill-current"><path d="M7.78 12.53a.75.75 0 0 1-1.06 0L2.47 8.28a.75.75 0 0 1 0-1.06l4.25-4.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042L4.81 7h7.44a.75.75 0 0 1 0 1.5H4.81l2.97 2.97a.75.75 0 0 1 0 1.06Z"></path></svg>
          Back to Repository
        </div>
        <h1 class="text-xl font-semibold ml-4">Pull Requests</h1>
      </div>
      <button id="new-pull-request-button" @click="goToNewPR" class="px-3 py-1 text-sm font-medium text-white bg-[#238636] rounded-md hover:bg-[#2ea44f]">
        New Pull Request
      </button>
    </header>

    <main class="max-w-6xl mx-auto p-6">
       <!-- Search and Filters -->
       <div class="bg-[#161b22] border border-gray-700 rounded-t-md p-4 flex flex-col md:flex-row gap-4 items-start md:items-center justify-between">
         <!-- Search -->
         <div class="flex-grow relative w-full md:w-auto">
            <input 
              id="pulls-search-input"
              type="text" 
              v-model="searchQuery"
              @keyup.enter="handleSearch"
              placeholder="is:pr is:open" 
              class="w-full px-3 py-1.5 bg-[#0d1117] border border-gray-600 rounded-md text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none pl-8"
            />
            <svg class="absolute left-2.5 top-2 w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
         </div>

         <!-- Filters -->
         <div class="flex gap-4 items-center flex-wrap">
            <!-- Open Filter Checkbox -->
            <label class="flex items-center space-x-2 text-sm text-gray-300 cursor-pointer select-none">
              <input 
                id="filter-open-pr-checkbox"
                type="checkbox" 
                class="form-checkbox h-4 w-4 text-blue-600 bg-gray-800 border-gray-600 rounded focus:ring-offset-gray-900"
                @change="handleFilterOpen"
                :checked="showOpenOnly"
              />
              <span>Open Only</span>
            </label>

             <!-- Reviews Slider -->
            <div class="flex items-center space-x-2">
                <label class="text-sm text-gray-400">Reviews > {{ reviewFilter }}</label>
                <input 
                    id="filter-reviews-slider"
                    type="range" 
                    min="0" 
                    max="20" 
                    step="1"
                    v-model.number="reviewFilter"
                    @input="handleReviewFilter"
                    class="w-32 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                />
            </div>

            <!-- Sort Dropdown -->
            <div class="relative">
                <button 
                    id="pulls-sort-dropdown"
                    @click="sortDropdownOpen = !sortDropdownOpen"
                    class="px-3 py-1.5 text-sm font-medium text-gray-300 bg-[#21262d] border border-gray-600 rounded-md hover:bg-gray-700 flex items-center gap-1"
                >
                    Sort <span class="text-xs">▾</span>
                </button>
                <div v-if="sortDropdownOpen" class="absolute right-0 z-10 mt-2 w-48 bg-[#161b22] border border-gray-700 rounded-md shadow-lg">
                    <div id="pulls-sort-newest" @click="handleSort('newest')" class="block px-4 py-2 text-sm text-gray-300 hover:bg-blue-600 hover:text-white cursor-pointer">Newest</div>
                    <div id="pulls-sort-oldest" @click="handleSort('oldest')" class="block px-4 py-2 text-sm text-gray-300 hover:bg-blue-600 hover:text-white cursor-pointer">Oldest</div>
                    <div id="pulls-sort-recently-updated" @click="handleSort('recently_updated')" class="block px-4 py-2 text-sm text-gray-300 hover:bg-blue-600 hover:text-white cursor-pointer">Recently updated</div>
                </div>
            </div>
         </div>
       </div>

       <!-- Pulls List -->
       <div id="pulls-list" class="border-x border-b border-gray-700 rounded-b-md divide-y divide-gray-700 bg-[#161b22]">
           <div 
             v-for="pr in filteredPulls" 
             :key="pr.id"
             :class="['p-4 hover:bg-[#1c2128] transition-colors flex gap-3', `data-id-${pr.id}`]"
           >
              <!-- Icon -->
              <div class="mt-1">
                  <svg v-if="pr.state === 'open'" aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-git-pull-request text-green-500"><path d="M1.5 3.25a2.25 2.25 0 1 1 3 2.122v5.256a2.251 2.251 0 1 1-1.5 0V5.372A2.25 2.25 0 0 1 1.5 3.25Zm5.677-.5a3.25 3.25 0 0 1 3.25 3.25v5.256a2.251 2.251 0 1 1-1.5 0V6a1.75 1.75 0 0 0-1.75-1.75H7.227ZM4.5 3.25a.75.75 0 1 0-1.5 0 .75.75 0 0 0 1.5 0ZM3.75 12a.75.75 0 1 0 0 1.5.75.75 0 0 0 0-1.5Zm9.75 0a.75.75 0 1 0 0 1.5.75.75 0 0 0 0-1.5Z"></path></svg>
                  <svg v-else-if="pr.state === 'merged'" aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-git-merge text-purple-500"><path d="M5.45 5.13a3.75 3.75 0 0 0-2.7 2.245 3.75 3.75 0 0 0 2.7 4.995V14a.75.75 0 0 0 1.5 0v-1.63a3.75 3.75 0 0 0 2.7-4.995 3.75 3.75 0 0 0-2.7-2.245V2.75a.75.75 0 0 0-1.5 0ZM11.28 2.22a.75.75 0 0 0-1.06 1.06L12.19 5.25H9.25a.75.75 0 0 0 0 1.5h2.94l-1.97 1.97a.75.75 0 1 0 1.06 1.06l3.25-3.25a.75.75 0 0 0 0-1.06Zm-4.58 7.53a2.25 2.25 0 1 1 0-4.5 2.25 2.25 0 0 1 0 4.5Z"></path></svg>
                  <svg v-else aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-git-pull-request-closed text-red-500"><path d="M1.5 3.25a2.25 2.25 0 1 1 3 2.122v5.256a2.251 2.251 0 1 1-1.5 0V5.372A2.25 2.25 0 0 1 1.5 3.25Zm5.677-.5a3.25 3.25 0 0 1 3.25 3.25v5.256a2.251 2.251 0 1 1-1.5 0V6a1.75 1.75 0 0 0-1.75-1.75H7.227ZM4.5 3.25a.75.75 0 1 0-1.5 0 .75.75 0 0 0 1.5 0ZM3.75 12a.75.75 0 1 0 0 1.5.75.75 0 0 0 0-1.5Zm9.75 0a.75.75 0 1 0 0 1.5.75.75 0 0 0 0-1.5Z"></path></svg>
              </div>

              <div class="flex-grow">
                  <h3 class="font-semibold text-white text-base hover:text-blue-400 cursor-pointer">
                      <span
                         :class="{
                           'pr-row-matched': isMatched(pr),
                           'pr-row-filtered': isFiltered(pr),
                           'pr-row-visible': !isMatched(pr) && !isFiltered(pr)
                         }"
                         @click="openPull(pr)"
                      >
                         {{ pr.title }}
                      </span>
                  </h3>
                  <div class="text-xs text-gray-400 mt-1">
                      #{{ pr.id.split('_')[1] }} opened on {{ pr.created_at }} by <span class="hover:text-blue-400 cursor-pointer">{{ pr.author_id }}</span>
                      · {{ pr.head }} into {{ pr.base }}
                  </div>
              </div>

              <!-- Reviews count -->
              <div class="flex items-start gap-1 text-gray-400 text-xs mt-1" v-if="pr.reviews > 0">
                  <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-comment-discussion"><path d="M1.75 1h8.5c.966 0 1.75.784 1.75 1.75v5.5A1.75 1.75 0 0 1 10.25 10H6v2.5a.75.75 0 0 1-1.28.53L1.75 10a.75.75 0 0 1-.75-.75v-6.5C1 1.784 1.784 1 2.75 1ZM2.5 2.75a.25.25 0 0 0-.25.25v6.5c0 .138.112.25.25.25h4.07l2.58 2.58V10.25a.75.75 0 0 1 .75-.75h.35a.25.25 0 0 0 .25-.25v-5.5a.25.25 0 0 0-.25-.25h-8.5Z"></path><path d="M13.25 5H14.5c.966 0 1.75.784 1.75 1.75v8.5A1.75 1.75 0 0 1 14.5 17h-4.07l-2.58-2.58V14.25a.75.75 0 0 1-.75-.75h-3.5a.25.25 0 0 1-.25-.25v-1.5a.75.75 0 0 0-1.5 0v1.5A1.75 1.75 0 0 0 3.5 15h2.94l2.28 2.28a.75.75 0 0 0 1.28-.53V15.5h4.5a.25.25 0 0 0 .25-.25v-8.5a.25.25 0 0 0-.25-.25h-1.25a.75.75 0 0 0 0 1.5Z"></path></svg>
                  {{ pr.reviews }}
              </div>
           </div>

            <!-- Empty State -->
            <div v-if="filteredPulls.length === 0" class="p-12 text-center">
                <h3 class="text-lg font-medium text-gray-300">No pull requests matched your search.</h3>
            </div>
       </div>
    </main>
  </div>
</template>

<script>
import { ref, computed } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';
import { FSMRuntime } from '../fsm/FSMRuntime';
import fsmData from '../../fsm.json';

export default {
  name: 'PULL_REQUESTS_LIST',
  setup() {
    const router = useRouter();
    const store = useSignatureStore();
    const dataStore = useDataStore();
    const fsmRuntime = new FSMRuntime(fsmData, { store, router });

    // State
    const searchQuery = ref('');
    const reviewFilter = ref(0);
    const showOpenOnly = ref(false);
    const sortDropdownOpen = ref(false);
    const currentSort = ref(null);

    // Computed
    const filteredPulls = computed(() => {
        let list = [...dataStore.pulls];
        const repoId = store.signature.repos_selected_repo_id;
        if (repoId) {
            list = list.filter(i => i.repo_id === repoId);
        }

        // Search
        if (searchQuery.value) {
            const q = searchQuery.value.toLowerCase();
            list = list.filter(i => i.title.toLowerCase().includes(q));
        }

        // Open Filter
        if (showOpenOnly.value) {
            list = list.filter(i => i.state === 'open');
        }

        // Reviews Filter
        if (reviewFilter.value > 0) {
            list = list.filter(i => i.reviews > reviewFilter.value);
        }

        // Sort
        list.sort((a, b) => {
            if (currentSort.value === 'oldest') return new Date(a.created_at) - new Date(b.created_at);
            // Default newest / recently updated
            return new Date(b.created_at) - new Date(a.created_at);
        });

        return list;
    });

    const isMatched = (pr) => store.signature.pulls_matched_pr_id === pr.id;
    const isFiltered = (pr) => store.signature.pulls_list_filters_applied === true;

    // Actions
    const goBack = async () => {
        store.setCurrentPageId('REPOSITORY_DETAIL');
        await router.push({ name: 'REPOSITORY_DETAIL' });
    };

    const goToNewPR = async () => {
        store.setCurrentPageId('NEW_PULL_REQUEST');
        await router.push({ name: 'NEW_PULL_REQUEST' });
    };

    const handleSearch = () => {
         const action = fsmData.pages.find(p => p.id === 'PULL_REQUESTS_LIST').actions.find(a => a.id === 'ACT_PULLS_SEARCH');
         if (action) {
             const match = filteredPulls.value.find(i => i.title.toLowerCase().includes(searchQuery.value.toLowerCase()));
             const params = { item_id: match ? match.id : null };
             const nextSig = fsmRuntime.applyEffects(action, store.signature, params);
             store.signature.pulls_matched_pr_id = nextSig.pulls_matched_pr_id;
             store.signature.pulls_has_searched = nextSig.pulls_has_searched;
         }
    };

    const updateFilter = (actionId, params = {}) => {
        const action = fsmData.pages.find(p => p.id === 'PULL_REQUESTS_LIST').actions.find(a => a.id === actionId);
        if (action) {
            const nextSig = fsmRuntime.applyEffects(action, store.signature, params);
            store.signature.pulls_list_filters_applied = nextSig.pulls_list_filters_applied;
        }
    };

    const handleFilterOpen = (e) => {
        showOpenOnly.value = e.target.checked;
        updateFilter('ACT_PULLS_FILTER_OPEN');
    };

    const handleReviewFilter = () => {
        updateFilter('ACT_PULLS_FILTER_REVIEWS_SLIDER', { widget: 'slider' });
    };

    const handleSort = (type) => {
        currentSort.value = type;
        sortDropdownOpen.value = false;
        updateFilter('ACT_PULLS_SORT', { widget: 'dropdown' });
    };

    const openPull = async (pr) => {
        let actionId = 'ACT_PULLS_OPEN_ANY';
        if (isMatched(pr)) actionId = 'ACT_PULLS_OPEN_MATCHED';
        else if (isFiltered(pr)) actionId = 'ACT_PULLS_OPEN_FILTERED';

        const action = fsmData.pages.find(p => p.id === 'PULL_REQUESTS_LIST').actions.find(a => a.id === actionId);
        if (action && fsmRuntime.checkPreconditions(action, store.signature)) {
            const params = { item_id: pr.id };
            const nextSig = fsmRuntime.applyEffects(action, store.signature, params);
            store.signature.pulls_selected_pr_id = nextSig.pulls_selected_pr_id;
            
            // Clean up
            if (nextSig.pulls_list_filters_applied === null) store.signature.pulls_list_filters_applied = null;
            if (nextSig.pulls_has_searched === null) store.signature.pulls_has_searched = null;
            if (nextSig.pulls_viewport_anchor_id === null) store.signature.pulls_viewport_anchor_id = null;

            store.setCurrentPageId('PULL_REQUEST_DETAIL');
            await router.push({ name: 'PULL_REQUEST_DETAIL', params: { item_id: pr.id } });
        } else {
             // Fallback
            store.signature.pulls_selected_pr_id = pr.id;
            store.setCurrentPageId('PULL_REQUEST_DETAIL');
            await router.push({ name: 'PULL_REQUEST_DETAIL', params: { item_id: pr.id } });
        }
    };

    return {
        searchQuery,
        reviewFilter,
        showOpenOnly,
        sortDropdownOpen,
        filteredPulls,
        handleSearch,
        handleFilterOpen,
        handleReviewFilter,
        handleSort,
        openPull,
        goBack,
        goToNewPR,
        isMatched,
        isFiltered
    };
  }
}
</script>