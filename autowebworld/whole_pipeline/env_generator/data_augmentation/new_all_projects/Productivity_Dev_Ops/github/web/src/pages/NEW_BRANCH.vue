<template>
  <div class="min-h-screen bg-[#0d1117] text-white font-sans flex flex-col items-center py-12">
    <div class="w-full max-w-xl px-6">
        <h1 class="text-2xl font-semibold mb-2">Create a branch</h1>
        <p class="text-gray-400 text-sm mb-6">Branch off from the main branch to work on your feature.</p>

        <div class="bg-[#161b22] border border-gray-700 rounded-md p-6 space-y-6">
            <!-- Name -->
            <div>
                <label class="block font-semibold mb-2">Branch name</label>
                <input 
                  id="new-branch-name"
                  type="text" 
                  v-model="branchName"
                  class="w-full px-3 py-2 bg-[#0d1117] border border-gray-600 rounded-md focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none"
                  placeholder="feature/my-new-branch"
                />
            </div>

            <!-- Source -->
            <div>
                <label class="block font-semibold mb-2">Source</label>
                <div class="relative">
                     <button id="new-branch-source-dropdown" @click="sourceDropdownOpen = !sourceDropdownOpen" class="w-full text-left px-3 py-2 bg-[#0d1117] border border-gray-600 rounded-md hover:border-gray-500 flex justify-between items-center">
                         {{ sourceBranch || 'Select source' }} <span>▾</span>
                     </button>
                     <div v-if="sourceDropdownOpen" class="absolute left-0 z-10 mt-1 w-full bg-[#161b22] border border-gray-700 rounded-md shadow-xl max-h-48 overflow-y-auto">
                         <div id="new-branch-source-main" @click="setSource('main')" class="px-4 py-2 hover:bg-blue-600 cursor-pointer text-sm">main</div>
                         <div id="new-branch-source-develop" @click="setSource('develop')" class="px-4 py-2 hover:bg-blue-600 cursor-pointer text-sm">develop</div>
                     </div>
                </div>
                <p class="text-xs text-gray-500 mt-2">
                    Your new branch will be based on <span class="font-mono text-gray-300">{{ sourceBranch || '...' }}</span>
                </p>
            </div>

            <hr class="border-gray-700" />

            <div class="flex justify-between items-center">
                 <button id="cancel-new-branch" @click="goBack" class="text-sm text-blue-400 hover:underline">
                    Cancel
                </button>
                <button 
                  id="create-branch-submit" 
                  @click="createBranch"
                  :disabled="!branchName || !sourceBranch"
                  :class="['px-4 py-2 text-white font-medium rounded-md', (branchName && sourceBranch) ? 'bg-[#238636] hover:bg-[#2ea44f]' : 'bg-gray-700 cursor-not-allowed opacity-50']"
                >
                    Create branch
                </button>
            </div>
        </div>
    </div>
  </div>
</template>

<script>
import { ref, watch } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { FSMRuntime } from '../fsm/FSMRuntime';
import fsmData from '../../fsm.json';

export default {
  name: 'NEW_BRANCH',
  setup() {
    const router = useRouter();
    const store = useSignatureStore();
    const fsmRuntime = new FSMRuntime(fsmData, { store, router });

    const branchName = ref('');
    const sourceBranch = ref('main');
    const sourceDropdownOpen = ref(false);

    // Sync
    watch(branchName, val => store.signature.new_branch_name = val);
    watch(sourceBranch, val => store.signature.new_branch_source = val);
    
    // Init defaults in store logic
    store.signature.new_branch_source = 'main';

    const setSource = (val) => {
        sourceBranch.value = val;
        sourceDropdownOpen.value = false;
    };

    const createBranch = async () => {
        const action = fsmData.pages.find(p => p.id === 'NEW_BRANCH').actions.find(a => a.id === 'ACT_NEW_BRANCH_CREATE');
        if (action && fsmRuntime.checkPreconditions(action, store.signature)) {
            store.setCurrentPageId('NEW_BRANCH_SUCCESS');
            await router.push({ name: 'NEW_BRANCH_SUCCESS' });
        }
    };

    const goBack = async () => {
        store.setCurrentPageId('BRANCHES_LIST');
        await router.push({ name: 'BRANCHES_LIST' });
    };

    return {
        branchName,
        sourceBranch,
        sourceDropdownOpen,
        setSource,
        createBranch,
        goBack
    };
  }
}
</script>