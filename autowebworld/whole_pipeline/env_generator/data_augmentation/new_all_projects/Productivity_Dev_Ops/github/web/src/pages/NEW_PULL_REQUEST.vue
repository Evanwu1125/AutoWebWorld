<template>
  <div class="min-h-screen bg-[#0d1117] text-white font-sans flex flex-col items-center py-12">
     <div class="w-full max-w-5xl px-6">
        <h1 class="text-2xl font-normal mb-6">Comparing changes</h1>
        
        <div class="bg-[#161b22] border border-gray-700 rounded-md p-4 mb-6 flex items-center gap-4 text-sm">
             <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4"></path></svg>
             
             <div class="flex items-center gap-2">
                 <span class="text-gray-400">base:</span>
                 <div class="relative">
                     <button id="base-branch-dropdown" @click="baseOpen = !baseOpen" class="px-3 py-1.5 bg-[#21262d] border border-gray-600 rounded-md hover:bg-gray-700 font-mono">
                         {{ baseBranch }} ▾
                     </button>
                     <div v-if="baseOpen" class="absolute left-0 z-10 mt-2 w-48 bg-[#161b22] border border-gray-700 rounded-md shadow-xl">
                         <div id="base-branch-main" @click="setBase('main')" class="px-4 py-2 hover:bg-blue-600 cursor-pointer">main</div>
                         <div id="base-branch-develop" @click="setBase('develop')" class="px-4 py-2 hover:bg-blue-600 cursor-pointer">develop</div>
                     </div>
                 </div>
             </div>

             <svg class="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>

             <div class="flex items-center gap-2">
                 <span class="text-gray-400">compare:</span>
                 <div class="relative">
                     <button id="compare-branch-dropdown" @click="compareOpen = !compareOpen" class="px-3 py-1.5 bg-[#21262d] border border-gray-600 rounded-md hover:bg-gray-700 font-mono">
                         {{ compareBranch }} ▾
                     </button>
                     <div v-if="compareOpen" class="absolute left-0 z-10 mt-2 w-48 bg-[#161b22] border border-gray-700 rounded-md shadow-xl">
                         <div id="compare-branch-feature" @click="setCompare('feature')" class="px-4 py-2 hover:bg-blue-600 cursor-pointer">feature</div>
                         <div id="compare-branch-bugfix" @click="setCompare('bugfix')" class="px-4 py-2 hover:bg-blue-600 cursor-pointer">bugfix</div>
                     </div>
                 </div>
             </div>
             
             <div class="ml-auto text-green-400 flex items-center gap-1" v-if="baseBranch && compareBranch">
                 <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
                 Able to merge
             </div>
        </div>

        <div class="flex flex-col md:flex-row gap-6">
            <!-- Form -->
            <div class="flex-grow bg-[#161b22] border border-gray-700 rounded-md">
                <div class="p-4 space-y-4">
                    <input 
                      id="pr_title"
                      type="text" 
                      v-model="prTitle"
                      placeholder="Title"
                      class="w-full px-3 py-2 bg-[#0d1117] border border-gray-600 rounded-md focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none font-semibold text-lg"
                    />

                    <textarea 
                      id="pr_body"
                      v-model="prBody"
                      placeholder="Leave a comment"
                      class="w-full bg-[#0d1117] border border-gray-600 rounded-md p-3 min-h-[200px] focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none"
                    ></textarea>
                </div>
                
                <div class="p-4 bg-[#161b22] border-t border-gray-700 flex justify-end items-center gap-4 rounded-b-md">
                     <button id="cancel-new-pr" @click="goBack" class="text-sm text-gray-400 hover:text-blue-400 hover:underline">
                        Cancel
                    </button>
                    <button 
                      id="submit-new-pr" 
                      @click="submitPR"
                      :disabled="!prTitle || !baseBranch || !compareBranch"
                      :class="['px-4 py-2 text-white font-medium rounded-md', (prTitle && baseBranch && compareBranch) ? 'bg-[#238636] hover:bg-[#2ea44f]' : 'bg-gray-700 cursor-not-allowed opacity-50']"
                    >
                        Create pull request
                    </button>
                </div>
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
  name: 'NEW_PULL_REQUEST',
  setup() {
    const router = useRouter();
    const store = useSignatureStore();
    const fsmRuntime = new FSMRuntime(fsmData, { store, router });

    const prTitle = ref('');
    const prBody = ref('');
    const baseBranch = ref('main');
    const compareBranch = ref('feature');
    const baseOpen = ref(false);
    const compareOpen = ref(false);

    // Sync
    watch(prTitle, val => store.signature.new_pr_title = val);
    watch(prBody, val => store.signature.new_pr_body = val);
    watch(baseBranch, val => store.signature.new_pr_base_branch = val);
    watch(compareBranch, val => store.signature.new_pr_compare_branch = val);

    // Initialize store defaults if needed (though signature.js has them null, UI sets them initially here for UX)
    store.signature.new_pr_base_branch = 'main';
    store.signature.new_pr_compare_branch = 'feature';

    const setBase = (val) => {
        baseBranch.value = val;
        baseOpen.value = false;
    };

    const setCompare = (val) => {
        compareBranch.value = val;
        compareOpen.value = false;
    };

    const submitPR = async () => {
        const action = fsmData.pages.find(p => p.id === 'NEW_PULL_REQUEST').actions.find(a => a.id === 'ACT_NEW_PR_SUBMIT');
        if (action && fsmRuntime.checkPreconditions(action, store.signature)) {
            store.setCurrentPageId('PR_CREATE_SUCCESS');
            await router.push({ name: 'PR_CREATE_SUCCESS' });
        }
    };

    const goBack = async () => {
        store.setCurrentPageId('PULL_REQUESTS_LIST');
        await router.push({ name: 'PULL_REQUESTS_LIST' });
    };

    return {
        prTitle,
        prBody,
        baseBranch,
        compareBranch,
        baseOpen,
        compareOpen,
        setBase,
        setCompare,
        submitPR,
        goBack
    };
  }
}
</script>