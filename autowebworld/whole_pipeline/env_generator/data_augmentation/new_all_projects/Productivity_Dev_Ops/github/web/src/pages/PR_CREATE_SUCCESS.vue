<template>
  <div class="min-h-screen bg-[#0d1117] text-white font-sans flex flex-col items-center justify-center py-12 px-6">
    <div class="max-w-md w-full text-center space-y-8">
        <div class="mx-auto w-24 h-24 bg-purple-900/30 rounded-full flex items-center justify-center">
             <svg class="w-12 h-12 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4"></path></svg>
        </div>
        
        <h1 class="text-3xl font-bold">Pull Request Created!</h1>
        <p class="text-gray-400">Your pull request has been successfully created.</p>

        <div class="flex flex-col space-y-3">
            <button 
                id="success-view-pr" 
                @click="viewPR"
                class="w-full px-4 py-2 text-white bg-[#238636] rounded-md font-medium hover:bg-[#2ea44f] transition-colors"
            >
                View Pull Request
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
  name: 'PR_CREATE_SUCCESS',
  setup() {
    const router = useRouter();
    const store = useSignatureStore();

    const viewPR = async () => {
        store.setCurrentPageId('PULL_REQUEST_DETAIL');
        await router.push({ name: 'PULL_REQUEST_DETAIL' });
    };

    const goHome = async () => {
        store.setCurrentPageId('HOME');
        await router.push({ name: 'HOME' });
    };

    return {
        viewPR,
        goHome
    };
  }
}
</script>