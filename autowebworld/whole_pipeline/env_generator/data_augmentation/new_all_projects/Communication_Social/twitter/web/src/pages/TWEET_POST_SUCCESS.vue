<template>
  <div class="flex flex-col items-center justify-center min-h-screen bg-black text-white p-6 text-center">
    <div class="w-16 h-16 bg-[#00BA7C] rounded-full flex items-center justify-center mb-6">
       <svg viewBox="0 0 24 24" aria-hidden="true" class="h-8 w-8 fill-white"><g><path d="M9 16.17l-4.17-4.17-1.42 1.42L9 19 21 7l-1.41-1.41z"></path></g></svg>
    </div>
    
    <h2 class="text-2xl font-bold mb-2">Post published!</h2>
    <p class="text-[#71767B] mb-8">Your post was sent successfully.</p>
    
    <div class="flex flex-col gap-4 w-full max-w-xs">
       <button id="success-view-tweet" @click="handleViewTweet" class="w-full border border-[#536471] text-white font-bold rounded-full py-3 hover:bg-white/10 transition-colors">
          View Post
       </button>
       <button id="success-go-home" @click="handleGoHome" class="w-full bg-white text-black font-bold rounded-full py-3 hover:bg-[#EFF3F4] transition-colors">
          Back to Home
       </button>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'TWEET_POST_SUCCESS',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();

    const handleViewTweet = () => {
       // Navigate to TWEET_DETAIL
       // In real app, we'd have the ID of created tweet. 
       // FSM says parameter tweet_id: {ITEM_ANY}. We'll mock 't11' (my new tweet)
       const newTweetId = 't11';
       signatureStore.selected_tweet_id = newTweetId;
       signatureStore.setCurrentPageId('TWEET_DETAIL');
       router.push({ name: 'TWEET_DETAIL', params: { tweet_id: newTweetId } });
    };

    const handleGoHome = () => {
       signatureStore.setCurrentPageId('HOME');
       router.push({ name: 'HOME' });
    };

    return {
       handleViewTweet,
       handleGoHome
    };
  }
}
</script>