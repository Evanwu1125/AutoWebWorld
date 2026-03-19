<template>
  <div class="min-h-screen bg-[#0d1117] text-white font-sans flex flex-col items-center justify-center py-12 px-6">
    <div class="max-w-md w-full text-center space-y-8">
        <div class="mx-auto w-24 h-24 bg-blue-900/30 rounded-full flex items-center justify-center">
             <svg class="w-12 h-12 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path></svg>
        </div>
        
        <h1 class="text-3xl font-bold">Profile Updated!</h1>
        <p class="text-gray-400">Your profile information has been saved.</p>

        <div class="flex flex-col space-y-3">
            <button 
                id="success-back-profile" 
                @click="viewProfile"
                class="w-full px-4 py-2 text-white bg-[#238636] rounded-md font-medium hover:bg-[#2ea44f] transition-colors"
            >
                Return to Profile
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
  name: 'PROFILE_UPDATE_SUCCESS',
  setup() {
    const router = useRouter();
    const store = useSignatureStore();

    const viewProfile = async () => {
        store.setCurrentPageId('PROFILE_OVERVIEW');
        await router.push({ name: 'PROFILE_OVERVIEW' });
    };

    const goHome = async () => {
        store.setCurrentPageId('HOME');
        await router.push({ name: 'HOME' });
    };

    return {
        viewProfile,
        goHome
    };
  }
}
</script>