<template>
  <div class="min-h-screen bg-[#0d1117] text-white font-sans">
    <!-- Header -->
    <header class="bg-[#161b22] border-b border-gray-700 py-4 px-6 flex items-center justify-between">
      <div class="flex items-center gap-4">
        <div id="branches-back-to-repo" class="cursor-pointer flex items-center gap-1 text-blue-400 hover:underline" @click="goBack">
          <svg height="16" aria-hidden="true" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-arrow-left fill-current"><path d="M7.78 12.53a.75.75 0 0 1-1.06 0L2.47 8.28a.75.75 0 0 1 0-1.06l4.25-4.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042L4.81 7h7.44a.75.75 0 0 1 0 1.5H4.81l2.97 2.97a.75.75 0 0 1 0 1.06Z"></path></svg>
          Back to Repository
        </div>
        <h1 class="text-xl font-semibold ml-4">Branches</h1>
      </div>
      <button id="new-branch-button" @click="goToNewBranch" class="px-3 py-1 text-sm font-medium text-white bg-[#238636] rounded-md hover:bg-[#2ea44f]">
        New Branch
      </button>
    </header>

    <main class="max-w-6xl mx-auto p-6">
        <!-- Filter Bar -->
        <div class="bg-[#161b22] border border-gray-700 rounded-t-md p-4 flex items-center justify-between">
            <h2 class="font-semibold text-gray-200">All Branches</h2>
            
            <div class="flex gap-4">
                <div class="relative">
                     <button id="branch-list-dropdown" @click="dropdownOpen = !dropdownOpen" class="px-3 py-1.5 bg-[#21262d] border border-gray-600 rounded-md text-sm font-medium text-gray-300 hover:bg-gray-700 flex items-center gap-2">
                         Filter: {{ selectedBranch || 'All' }} ▾
                     </button>
                     <div v-if="dropdownOpen" class="absolute right-0 z-10 mt-2 w-48 bg-[#161b22] border border-gray-700 rounded-md shadow-xl">
                         <div id="branch-main" @click="filterBranch('main')" class="px-4 py-2 text-sm hover:bg-blue-600 cursor-pointer">main</div>
                         <div id="branch-develop" @click="filterBranch('develop')" class="px-4 py-2 text-sm hover:bg-blue-600 cursor-pointer">develop</div>
                         <div id="branch-feature" @click="filterBranch('feature')" class="px-4 py-2 text-sm hover:bg-blue-600 cursor-pointer">feature</div>
                     </div>
                </div>

                <button id="compare-branches-link" @click="goToCompare" class="px-3 py-1.5 bg-[#21262d] border border-gray-600 rounded-md text-sm font-medium text-gray-300 hover:bg-gray-700">
                    Compare
                </button>
            </div>
        </div>

        <!-- Branch List -->
        <div class="border-x border-b border-gray-700 rounded-b-md bg-[#0d1117]">
            <div v-for="branch in filteredBranches" :key="branch.id" class="p-4 border-b border-gray-800 last:border-0 flex items-center justify-between hover:bg-[#161b22]">
                <div>
                    <h3 class="font-semibold text-blue-400 flex items-center gap-2">
                        {{ branch.name }}
                        <span v-if="branch.name === 'main'" class="px-2 py-0.5 border border-gray-600 rounded-full text-xs text-gray-400">Default</span>
                         <span v-if="branch.protected" class="px-2 py-0.5 border border-green-800 bg-green-900/20 text-green-400 rounded-full text-xs">Protected</span>
                    </h3>
                    <div class="text-xs text-gray-500 mt-1">
                        Updated 2 days ago by user
                    </div>
                </div>
                <div class="flex items-center gap-4 text-sm text-gray-400">
                     <div class="flex items-center gap-1">
                         <div class="w-24 h-2 bg-gray-700 rounded-full overflow-hidden">
                             <div class="h-full bg-green-500" style="width: 60%"></div>
                         </div>
                     </div>
                     <button class="hover:text-red-400">
                         <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path></svg>
                     </button>
                </div>
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
  name: 'BRANCHES_LIST',
  setup() {
    const router = useRouter();
    const store = useSignatureStore();
    const dataStore = useDataStore();
    const fsmRuntime = new FSMRuntime(fsmData, { store, router });

    const dropdownOpen = ref(false);
    const selectedBranch = ref(''); // Local state for UI filtering feedback if needed, or mapped from signature

    const filteredBranches = computed(() => {
        let list = dataStore.branches;
        // If signature has a filter, apply it
        if (store.signature.branches_selected_branch_name) {
             // The FSM action ACT_BRANCHES_SELECT_BRANCH sets branches_selected_branch_name
             // But usually filtering "all branches" list by one branch is just search. 
             // Assuming the requirement implies filtering the list view:
             // Check if filter value is valid
             if (store.signature.branches_selected_branch_name && store.signature.branches_selected_branch_name !== 'all') {
                  list = list.filter(b => b.name.includes(store.signature.branches_selected_branch_name));
             }
        }
        return list;
    });

    const goBack = async () => {
        store.setCurrentPageId('REPOSITORY_DETAIL');
        await router.push({ name: 'REPOSITORY_DETAIL' });
    };

    const goToNewBranch = async () => {
        store.setCurrentPageId('NEW_BRANCH');
        await router.push({ name: 'NEW_BRANCH' });
    };

    const goToCompare = async () => {
         const action = fsmData.pages.find(p => p.id === 'BRANCHES_LIST').actions.find(a => a.id === 'ACT_BRANCHES_GO_COMPARE');
         if (action && fsmRuntime.checkPreconditions(action, store.signature)) {
             store.setCurrentPageId('COMPARE_BRANCHES');
             await router.push({ name: 'COMPARE_BRANCHES' });
         } else {
             // Just go for robustness
             store.setCurrentPageId('COMPARE_BRANCHES');
             await router.push({ name: 'COMPARE_BRANCHES' });
         }
    };

    const filterBranch = (name) => {
        const action = fsmData.pages.find(p => p.id === 'BRANCHES_LIST').actions.find(a => a.id === 'ACT_BRANCHES_SELECT_BRANCH');
        if (action) {
            // Apply effect directly to store
            store.signature.branches_selected_branch_name = name;
            selectedBranch.value = name;
            dropdownOpen.value = false;
        }
    };

    return {
        dropdownOpen,
        selectedBranch,
        filteredBranches,
        goBack,
        goToNewBranch,
        goToCompare,
        filterBranch
    };
  }
}
</script>