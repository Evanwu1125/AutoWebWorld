<template>
  <div class="min-h-screen bg-[#0d1117] text-white font-sans flex flex-col items-center justify-center py-12 px-6">
    <div class="max-w-md w-full text-center space-y-8">
        <div class="mx-auto w-24 h-24 bg-blue-900/30 rounded-full flex items-center justify-center">
             <svg class="w-12 h-12 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"></path></svg>
        </div>
        
        <h1 class="text-3xl font-bold">Branch Created!</h1>
        <p class="text-gray-400">You can now switch to this branch and start working.</p>

        <div class="flex flex-col space-y-3">
            <button 
                id="success-view-branches" 
                @click="viewBranches"
                class="w-full px-4 py-2 text-white bg-[#238636] rounded-md font-medium hover:bg-[#2ea44f] transition-colors"
            >
                View Branches
            </button>
            <button 
                id="success-go-home" 
                @click="goHome"
                class="w-full px-4 py-2 text-gray-300 bg-[#21262d] border border-gray-600 rounded-md font-medium hover:bg-gray-700 transition-colors"
            >
                Back to Home
            </button>
        </div>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'NEW_BRANCH_SUCCESS',
  setup() {
    const router = useRouter();
    const store = useSignatureStore();

    const viewBranches = async () => {
        store.setCurrentPageId('BRANCHES_LIST');
        await router.push({ name: 'BRANCHES_LIST' });
    };

    const goHome = async () => {
        store.setCurrentPageId('HOME');
        await router.push({ name: 'HOME' });
    };

    return {
        viewBranches,
        goHome
    };
  }
}
</script>