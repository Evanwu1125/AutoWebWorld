<template>
  <div class="flex flex-col items-center justify-center min-h-screen bg-black text-white p-6 text-center">
    <div class="w-16 h-16 bg-[#1D9BF0] rounded-full flex items-center justify-center mb-6">
       <svg viewBox="0 0 24 24" aria-hidden="true" class="h-8 w-8 fill-white"><g><path d="M6 3V2h2v1h6V2h2v1h1.5C18.88 3 20 4.12 20 5.5v13c0 1.38-1.12 2.5-2.5 2.5h-13C3.12 21 2 19.88 2 18.5v-13C2 4.12 3.12 3 4.5 3H6zm9.5 8h-2v2h2v-2zm-5 0h-2v2h2v-2zm-5 0h-2v2h2v-2zm10 4h-2v2h2v-2zm-5 0h-2v2h2v-2zm-5 0h-2v2h2v-2z"></path></g></svg>
    </div>
    
    <h2 class="text-2xl font-bold mb-2">Post Scheduled</h2>
    <p class="text-[#71767B] mb-8">Your post will be sent on the selected date.</p>
    
    <div class="flex flex-col gap-4 w-full max-w-xs">
       <button id="schedule-success-view-queue" @click="handleViewQueue" class="w-full border border-[#536471] text-white font-bold rounded-full py-3 hover:bg-white/10 transition-colors">
          View Scheduled Posts
       </button>
       <button id="schedule-success-go-home" @click="handleGoHome" class="w-full bg-white text-black font-bold rounded-full py-3 hover:bg-[#EFF3F4] transition-colors">
          Back to Home
       </button>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'TWEET_SCHEDULE_SUCCESS',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();

    const handleViewQueue = () => {
       // FSM: ACT_TWEET_SCHEDULE_SUCCESS_VIEW_SCHEDULED -> PROFILE_TWEETS
       signatureStore.setCurrentPageId('PROFILE_TWEETS');
       router.push({ name: 'PROFILE_TWEETS' });
    };

    const handleGoHome = () => {
       signatureStore.setCurrentPageId('HOME');
       router.push({ name: 'HOME' });
    };

    return {
       handleViewQueue,
       handleGoHome
    };
  }
}
</script>