<template>
  <div class="min-h-screen bg-[#0d1117] text-white font-sans">
    <!-- Header -->
    <header class="bg-[#161b22] border-b border-gray-700 py-4 px-6 flex items-center justify-between">
      <div class="flex items-center gap-4">
        <div id="issues-back-to-repo" class="cursor-pointer flex items-center gap-1 text-blue-400 hover:underline" @click="goBack">
          <svg height="16" aria-hidden="true" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-arrow-left fill-current"><path d="M7.78 12.53a.75.75 0 0 1-1.06 0L2.47 8.28a.75.75 0 0 1 0-1.06l4.25-4.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042L4.81 7h7.44a.75.75 0 0 1 0 1.5H4.81l2.97 2.97a.75.75 0 0 1 0 1.06Z"></path></svg>
          Back to Repository
        </div>
        <h1 class="text-xl font-semibold ml-4">Issues</h1>
      </div>
      <button id="new-issue-button" @click="goToNewIssue" class="px-3 py-1 text-sm font-medium text-white bg-[#238636] rounded-md hover:bg-[#2ea44f]">
        New Issue
      </button>
    </header>

    <main class="max-w-6xl mx-auto p-6">
       <!-- Search and Filters -->
       <div class="bg-[#161b22] border border-gray-700 rounded-t-md p-4 flex flex-col md:flex-row gap-4 items-start md:items-center justify-between">
         <!-- Search -->
         <div class="flex-grow relative w-full md:w-auto">
            <input 
              id="issues-search-input"
              type="text" 
              v-model="searchQuery"
              @keyup.enter="handleSearch"
              placeholder="is:issue is:open" 
              class="w-full px-3 py-1.5 bg-[#0d1117] border border-gray-600 rounded-md text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none pl-8"
            />
            <svg class="absolute left-2.5 top-2 w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
         </div>

         <!-- Filters -->
         <div class="flex gap-4 items-center flex-wrap">
            <!-- Open Filter Checkbox -->
            <label class="flex items-center space-x-2 text-sm text-gray-300 cursor-pointer select-none">
              <input 
                id="filter-open-checkbox"
                type="checkbox" 
                class="form-checkbox h-4 w-4 text-blue-600 bg-gray-800 border-gray-600 rounded focus:ring-offset-gray-900"
                @change="handleFilterOpen"
                :checked="showOpenOnly"
              />
              <span>Open Only</span>
            </label>

             <!-- Comments Slider -->
            <div class="flex items-center space-x-2">
                <label class="text-sm text-gray-400">Comments > {{ commentFilter }}</label>
                <input 
                    id="filter-comments-slider"
                    type="range" 
                    min="0" 
                    max="50" 
                    step="1"
                    v-model.number="commentFilter"
                    @input="handleCommentFilter"
                    class="w-32 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                />
            </div>

            <!-- Sort Dropdown -->
            <div class="relative">
                <button 
                    id="issues-sort-dropdown"
                    @click="sortDropdownOpen = !sortDropdownOpen"
                    class="px-3 py-1.5 text-sm font-medium text-gray-300 bg-[#21262d] border border-gray-600 rounded-md hover:bg-gray-700 flex items-center gap-1"
                >
                    Sort <span class="text-xs">▾</span>
                </button>
                <div v-if="sortDropdownOpen" class="absolute right-0 z-10 mt-2 w-48 bg-[#161b22] border border-gray-700 rounded-md shadow-lg">
                    <div id="issues-sort-newest" @click="handleSort('newest')" class="block px-4 py-2 text-sm text-gray-300 hover:bg-blue-600 hover:text-white cursor-pointer">Newest</div>
                    <div id="issues-sort-oldest" @click="handleSort('oldest')" class="block px-4 py-2 text-sm text-gray-300 hover:bg-blue-600 hover:text-white cursor-pointer">Oldest</div>
                    <div id="issues-sort-most-commented" @click="handleSort('most_commented')" class="block px-4 py-2 text-sm text-gray-300 hover:bg-blue-600 hover:text-white cursor-pointer">Most commented</div>
                </div>
            </div>
         </div>
       </div>

       <!-- Issues List -->
       <div id="issues-list" class="border-x border-b border-gray-700 rounded-b-md divide-y divide-gray-700 bg-[#161b22]">
           <div 
             v-for="issue in filteredIssues" 
             :key="issue.id"
             :class="['p-4 hover:bg-[#1c2128] transition-colors flex gap-3', `data-id-${issue.id}`]"
           >
              <!-- Icon -->
              <div class="mt-1">
                  <svg v-if="issue.state === 'open'" aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-issue-opened text-green-500"><path d="M8 9.5a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3Z"></path><path d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0ZM1.5 8a6.5 6.5 0 1 0 13 0 6.5 6.5 0 0 0-13 0Z"></path></svg>
                  <svg v-else aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-issue-closed text-purple-500"><path d="M11.28 6.78a.75.75 0 0 0-1.06-1.06L7.25 8.69 5.78 7.22a.75.75 0 0 0-1.06 1.06l2 2a.75.75 0 0 0 1.06 0l3.5-3.5Z"></path><path d="M16 8A8 8 0 1 1 0 8a8 8 0 0 1 16 0Zm-1.5 0a6.5 6.5 0 1 0-13 0 6.5 6.5 0 0 0 13 0Z"></path></svg>
              </div>

              <div class="flex-grow">
                  <h3 class="font-semibold text-white text-base hover:text-blue-400 cursor-pointer">
                      <span
                         :class="{
                           'issue-row-matched': isMatched(issue),
                           'issue-row-filtered': isFiltered(issue),
                           'issue-row-visible': !isMatched(issue) && !isFiltered(issue)
                         }"
                         @click="openIssue(issue)"
                      >
                         {{ issue.title }}
                      </span>
                      <span v-for="label in issue.labels" :key="label" 
                        :class="['ml-2 px-2 py-0.5 rounded-full text-xs font-medium border', getLabelClass(label)]">
                        {{ label }}
                      </span>
                  </h3>
                  <div class="text-xs text-gray-400 mt-1">
                      #{{ issue.id.split('_')[1] }} opened on {{ issue.created_at }} by <span class="hover:text-blue-400 cursor-pointer">{{ issue.author_id }}</span>
                  </div>
              </div>

              <!-- Comments count -->
              <div class="flex items-start gap-1 text-gray-400 text-xs mt-1" v-if="issue.comments > 0">
                  <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-comment"><path d="M1 2.75C1 1.784 1.784 1 2.75 1h10.5c.966 0 1.75.784 1.75 1.75v7.5A1.75 1.75 0 0 1 13.25 12H9.06l-2.573 2.573A1.458 1.458 0 0 1 4 13.543V12H2.75A1.75 1.75 0 0 1 1 10.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h2a.75.75 0 0 1 .75.75v2.19l2.72-2.72a.75.75 0 0 1 .53-.22h4.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path></svg>
                  {{ issue.comments }}
              </div>
           </div>

            <!-- Empty State -->
            <div v-if="filteredIssues.length === 0" class="p-12 text-center">
                <h3 class="text-lg font-medium text-gray-300">No issues matched your search.</h3>
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
  name: 'ISSUES_LIST',
  setup() {
    const router = useRouter();
    const store = useSignatureStore();
    const dataStore = useDataStore();
    const fsmRuntime = new FSMRuntime(fsmData, { store, router });

    // State
    const searchQuery = ref('');
    const commentFilter = ref(0);
    const showOpenOnly = ref(false);
    const sortDropdownOpen = ref(false);
    const currentSort = ref(null);

    // Computed
    const filteredIssues = computed(() => {
        let list = [...dataStore.issues];
        // Filter by repo context if needed (assume all for now or filter by selected repo if we had that context persisted cleanly)
        // Since the store has repos_selected_repo_id, we should filter by it
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

        // Comments Filter
        if (commentFilter.value > 0) {
            list = list.filter(i => i.comments > commentFilter.value);
        }

        // Sort
        list.sort((a, b) => {
            if (currentSort.value === 'most_commented') return b.comments - a.comments;
            if (currentSort.value === 'oldest') return new Date(a.created_at) - new Date(b.created_at);
            // Default newest
            return new Date(b.created_at) - new Date(a.created_at);
        });

        return list;
    });

    // Helper
    const getLabelClass = (label) => {
        if (label === 'bug') return 'border-red-500 text-red-500';
        if (label === 'enhancement') return 'border-blue-500 text-blue-500';
        if (label === 'question') return 'border-purple-500 text-purple-500';
        return 'border-gray-500 text-gray-500';
    };

    const isMatched = (issue) => store.signature.issues_matched_issue_id === issue.id;
    const isFiltered = (issue) => store.signature.issues_list_filters_applied === true;

    // Actions
    const goBack = async () => {
        store.setCurrentPageId('REPOSITORY_DETAIL');
        await router.push({ name: 'REPOSITORY_DETAIL' });
    };

    const goToNewIssue = async () => {
        store.setCurrentPageId('NEW_ISSUE');
        await router.push({ name: 'NEW_ISSUE' });
    };

    const handleSearch = () => {
         const action = fsmData.pages.find(p => p.id === 'ISSUES_LIST').actions.find(a => a.id === 'ACT_ISSUES_SEARCH');
         if (action) {
             const match = filteredIssues.value.find(i => i.title.toLowerCase().includes(searchQuery.value.toLowerCase()));
             const params = { item_id: match ? match.id : null };
             const nextSig = fsmRuntime.applyEffects(action, store.signature, params);
             store.signature.issues_matched_issue_id = nextSig.issues_matched_issue_id;
             store.signature.issues_has_searched = nextSig.issues_has_searched;
         }
    };

    const updateFilter = (actionId, params = {}) => {
        const action = fsmData.pages.find(p => p.id === 'ISSUES_LIST').actions.find(a => a.id === actionId);
        if (action) {
            const nextSig = fsmRuntime.applyEffects(action, store.signature, params);
            store.signature.issues_list_filters_applied = nextSig.issues_list_filters_applied;
        }
    };

    const handleFilterOpen = (e) => {
        showOpenOnly.value = e.target.checked;
        updateFilter('ACT_ISSUES_FILTER_OPEN');
    };

    const handleCommentFilter = () => {
        updateFilter('ACT_ISSUES_FILTER_COMMENTS_SLIDER', { widget: 'slider' });
    };

    const handleSort = (type) => {
        currentSort.value = type;
        sortDropdownOpen.value = false;
        updateFilter('ACT_ISSUES_SORT', { widget: 'dropdown' });
    };

    const openIssue = async (issue) => {
        let actionId = 'ACT_ISSUES_OPEN_ANY';
        if (isMatched(issue)) actionId = 'ACT_ISSUES_OPEN_MATCHED';
        else if (isFiltered(issue)) actionId = 'ACT_ISSUES_OPEN_FILTERED';

        const action = fsmData.pages.find(p => p.id === 'ISSUES_LIST').actions.find(a => a.id === actionId);
        if (action && fsmRuntime.checkPreconditions(action, store.signature)) {
            const params = { item_id: issue.id };
            const nextSig = fsmRuntime.applyEffects(action, store.signature, params);
            store.signature.issues_selected_issue_id = nextSig.issues_selected_issue_id;
            
            // Clean up
            if (nextSig.issues_list_filters_applied === null) store.signature.issues_list_filters_applied = null;
            if (nextSig.issues_has_searched === null) store.signature.issues_has_searched = null;
            if (nextSig.issues_viewport_anchor_id === null) store.signature.issues_viewport_anchor_id = null;

            store.setCurrentPageId('ISSUE_DETAIL');
            await router.push({ name: 'ISSUE_DETAIL', params: { item_id: issue.id } });
        } else {
            // Fallback
            store.signature.issues_selected_issue_id = issue.id;
            store.setCurrentPageId('ISSUE_DETAIL');
            await router.push({ name: 'ISSUE_DETAIL', params: { item_id: issue.id } });
        }
    };

    return {
        searchQuery,
        commentFilter,
        showOpenOnly,
        sortDropdownOpen,
        filteredIssues,
        handleSearch,
        handleFilterOpen,
        handleCommentFilter,
        handleSort,
        openIssue,
        goBack,
        goToNewIssue,
        getLabelClass,
        isMatched,
        isFiltered
    };
  }
}
</script>