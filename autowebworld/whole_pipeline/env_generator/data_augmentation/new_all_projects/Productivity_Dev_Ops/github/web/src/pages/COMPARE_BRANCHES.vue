<template>
  <div class="min-h-screen bg-[#0d1117] text-white font-sans">
    <!-- Header -->
    <header class="bg-[#161b22] border-b border-gray-700 py-4 px-6 flex items-center gap-4">
      <div id="compare-back-to-branches" class="cursor-pointer flex items-center gap-1 text-blue-400 hover:underline" @click="goBack">
          <svg height="16" aria-hidden="true" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-arrow-left fill-current"><path d="M7.78 12.53a.75.75 0 0 1-1.06 0L2.47 8.28a.75.75 0 0 1 0-1.06l4.25-4.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042L4.81 7h7.44a.75.75 0 0 1 0 1.5H4.81l2.97 2.97a.75.75 0 0 1 0 1.06Z"></path></svg>
          Back to Branches
      </div>
      <h1 class="text-xl font-semibold ml-4">Comparing changes</h1>
    </header>

    <main class="max-w-5xl mx-auto p-6">
        <div class="bg-[#161b22] border border-gray-700 rounded-md p-4 mb-6 flex flex-col md:flex-row items-center gap-4 text-sm">
             <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4"></path></svg>
             
             <div class="flex items-center gap-2">
                 <span class="text-gray-400">base:</span>
                 <div class="relative">
                     <button id="compare-base-dropdown" @click="baseOpen = !baseOpen" class="px-3 py-1.5 bg-[#21262d] border border-gray-600 rounded-md hover:bg-gray-700 font-mono">
                         {{ baseBranch }} ▾
                     </button>
                     <div v-if="baseOpen" class="absolute left-0 z-10 mt-2 w-48 bg-[#161b22] border border-gray-700 rounded-md shadow-xl">
                         <div id="compare-base-main" @click="setBase('main')" class="px-4 py-2 hover:bg-blue-600 cursor-pointer">main</div>
                         <div id="compare-base-develop" @click="setBase('develop')" class="px-4 py-2 hover:bg-blue-600 cursor-pointer">develop</div>
                     </div>
                 </div>
             </div>

             <svg class="w-4 h-4 text-gray-500 hidden md:block" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>

             <div class="flex items-center gap-2">
                 <span class="text-gray-400">compare:</span>
                 <div class="relative">
                     <button id="compare-head-dropdown" @click="headOpen = !headOpen" class="px-3 py-1.5 bg-[#21262d] border border-gray-600 rounded-md hover:bg-gray-700 font-mono">
                         {{ headBranch }} ▾
                     </button>
                     <div v-if="headOpen" class="absolute left-0 z-10 mt-2 w-48 bg-[#161b22] border border-gray-700 rounded-md shadow-xl">
                         <div id="compare-head-feature" @click="setHead('feature')" class="px-4 py-2 hover:bg-blue-600 cursor-pointer">feature</div>
                         <div id="compare-head-bugfix" @click="setHead('bugfix')" class="px-4 py-2 hover:bg-blue-600 cursor-pointer">bugfix</div>
                     </div>
                 </div>
             </div>
             
             <button id="view-changes-button" @click="viewChanges" class="ml-auto px-4 py-1.5 bg-[#238636] text-white font-medium rounded-md hover:bg-[#2ea44f]">
                 View Changes
             </button>
        </div>

        <div v-if="showChanges" class="space-y-6">
            <div class="border border-gray-700 rounded-md bg-[#0d1117]">
                <div class="bg-[#161b22] px-4 py-2 border-b border-gray-700 font-mono text-sm flex justify-between">
                     <span>src/main.js</span>
                     <span class="text-green-400">+12</span>
                </div>
                <div class="p-4 overflow-x-auto">
                    <pre class="text-sm"><code class="language-javascript"><span class="text-gray-500">  const app = createApp(App)</span>
<span class="text-green-400">+ app.use(createPinia())</span>
<span class="text-green-400">+ app.use(router)</span>
<span class="text-gray-500">  app.mount('#app')</span></code></pre>
                </div>
            </div>

             <div class="border border-gray-700 rounded-md bg-[#0d1117]">
                <div class="bg-[#161b22] px-4 py-2 border-b border-gray-700 font-mono text-sm flex justify-between">
                     <span>package.json</span>
                     <span class="text-green-400">+2</span>
                </div>
                <div class="p-4 overflow-x-auto">
                    <pre class="text-sm"><code class="language-json"><span class="text-gray-500">  "dependencies": {</span>
<span class="text-green-400">+   "pinia": "^2.1.7",</span>
<span class="text-green-400">+   "vue-router": "^4.2.5"</span>
<span class="text-gray-500">  }</span></code></pre>
                </div>
            </div>
        </div>

        <div v-else class="text-center py-12 text-gray-500">
            Select branches and click "View Changes" to see the diff.
        </div>

    </main>
  </div>
</template>

<script>
import { ref, watch } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { FSMRuntime } from '../fsm/FSMRuntime';
import fsmData from '../../fsm.json';

export default {
  name: 'COMPARE_BRANCHES',
  setup() {
    const router = useRouter();
    const store = useSignatureStore();
    const fsmRuntime = new FSMRuntime(fsmData, { store, router });

    const baseBranch = ref('main');
    const headBranch = ref('feature');
    const baseOpen = ref(false);
    const headOpen = ref(false);
    const showChanges = ref(false);

    // Sync
    watch(baseBranch, val => store.signature.compare_base = val);
    watch(headBranch, val => store.signature.compare_head = val);

    // Defaults
    store.signature.compare_base = 'main';
    store.signature.compare_head = 'feature';

    const setBase = (val) => {
        baseBranch.value = val;
        baseOpen.value = false;
        showChanges.value = false;
    };

    const setHead = (val) => {
        headBranch.value = val;
        headOpen.value = false;
        showChanges.value = false;
    };

    const viewChanges = () => {
         const action = fsmData.pages.find(p => p.id === 'COMPARE_BRANCHES').actions.find(a => a.id === 'ACT_COMPARE_VIEW_CHANGES');
         if (action && fsmRuntime.checkPreconditions(action, store.signature)) {
             showChanges.value = true;
         }
    };

    const goBack = async () => {
        store.setCurrentPageId('BRANCHES_LIST');
        await router.push({ name: 'BRANCHES_LIST' });
    };

    return {
        baseBranch,
        headBranch,
        baseOpen,
        headOpen,
        showChanges,
        setBase,
        setHead,
        viewChanges,
        goBack
    };
  }
}
</script>