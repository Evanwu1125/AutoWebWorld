<template>
  <div class="flex flex-col min-h-screen bg-black text-white">
    <!-- Header -->
    <div class="sticky top-0 z-30 bg-black/80 backdrop-blur-md border-b border-[#2F3336] px-4 py-3 flex items-center gap-4">
      <div id="back-timeline" @click="handleBack" class="p-2 rounded-full hover:bg-white/10 cursor-pointer transition-colors">
         <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 fill-current"><g><path d="M7.414 13l5.043 5.04-1.414 1.42L3.586 12l7.457-7.46 1.414 1.42L7.414 11H21v2H7.414z"></path></g></svg>
      </div>
      <h2 class="text-xl font-bold">Post</h2>
    </div>

    <!-- Tweet Content -->
    <div v-if="tweet" class="p-4 border-b border-[#2F3336]">
       <!-- Author Info -->
       <div class="flex gap-3 mb-4">
          <div class="w-12 h-12 rounded-full overflow-hidden bg-gray-700 cursor-pointer" id="tweet-author-handle" @click="handleOpenProfile(author?.id)">
             <img :src="author?.avatar || '/images/photo1766328458.jpg'" alt="avatar" class="w-full h-full object-cover">
          </div>
          <div class="flex flex-col">
             <div class="font-bold text-white hover:underline cursor-pointer" @click="handleOpenProfile(author?.id)">
                {{ author?.name }}
             </div>
             <div class="text-[#71767B]">
                {{ author?.handle }}
             </div>
          </div>
       </div>

       <!-- Text -->
       <div class="text-xl leading-8 whitespace-pre-wrap break-words mb-4">
          {{ tweet.content }}
       </div>

       <!-- Media -->
       <div v-if="tweet.has_media && tweet.media_url" class="mb-4 rounded-2xl overflow-hidden border border-[#2F3336]">
          <img :src="tweet.media_url" alt="media" class="w-full object-cover">
       </div>

       <!-- Timestamp & Views -->
       <div class="text-[#71767B] text-[15px] py-4 border-b border-[#2F3336]">
          <span class="hover:underline cursor-pointer">10:30 AM · Oct 22, 2025</span>
          <span class="mx-1">·</span>
          <span class="text-white font-bold">{{ tweet.views }}</span>
          <span class="ml-1">Views</span>
       </div>

       <!-- Engagement Stats -->
       <div class="flex gap-6 py-4 border-b border-[#2F3336] text-[#71767B] text-[15px]">
          <div><span class="font-bold text-white">{{ tweet.retweets }}</span> Retweets</div>
          <div><span class="font-bold text-white">{{ tweet.replies }}</span> Quotes</div>
          <div><span class="font-bold text-white">{{ tweet.likes }}</span> Likes</div>
       </div>

       <!-- Interaction Buttons -->
       <div class="flex justify-around py-3 border-b border-[#2F3336]">
          <div id="tweet-reply-button" @click="handleReply" class="group p-2 cursor-pointer text-[#71767B] hover:text-[#1D9BF0]">
             <svg viewBox="0 0 24 24" aria-hidden="true" class="h-6 w-6 fill-current"><g><path d="M1.751 10c0-4.42 3.584-8 8.005-8h4.366c4.49 0 8.129 3.64 8.129 8.13 0 2.96-1.607 5.68-4.196 7.11l-8.054 4.46v-3.69h-.067c-4.49.1-8.183-3.51-8.183-8.01zm8.005-6c-3.317 0-6.005 2.69-6.005 6 0 3.37 2.77 6.08 6.138 6.01l.351-.01h1.761v2.3l5.087-2.81c1.951-1.08 3.163-3.13 3.163-5.36 0-3.39-2.744-6.13-6.129-6.13H9.756z"></path></g></svg>
          </div>
          <div class="group p-2 cursor-pointer text-[#71767B] hover:text-[#00BA7C]">
             <svg viewBox="0 0 24 24" aria-hidden="true" class="h-6 w-6 fill-current"><g><path d="M4.5 3.88l4.432 4.14-1.364 1.46L5.5 7.55V16c0 1.1.896 2 2 2H13v2H7.5c-2.209 0-4-1.79-4-4V7.55L1.432 9.48.068 8.02 4.5 3.88zM16.5 6H11V4h5.5c2.209 0 4 1.79 4 4v8.45l2.068-1.93 1.364 1.46-4.432 4.14-4.432-4.14 1.364-1.46 2.068 1.93V8c0-1.1-.896-2-2-2z"></path></g></svg>
          </div>
          <div class="group p-2 cursor-pointer text-[#71767B] hover:text-[#F91880]">
             <svg viewBox="0 0 24 24" aria-hidden="true" class="h-6 w-6 fill-current"><g><path d="M16.697 5.5c-1.222-.06-2.679.51-3.89 2.16l-.805 1.09-.806-1.09C9.984 6.01 8.526 5.44 7.304 5.5c-1.243.07-2.349.78-2.91 1.91-.552 1.12-.633 2.78.479 4.82 1.074 1.97 3.257 4.27 7.129 6.61 3.87-2.34 6.052-4.64 7.126-6.61 1.111-2.04 1.03-3.7.477-4.82-.561-1.13-1.666-1.84-2.908-1.91zm4.187 7.69c-1.351 2.48-4.001 5.12-8.379 7.67l-.503.3-.504-.3c-4.379-2.55-7.029-5.19-8.382-7.67-1.36-2.5-1.41-4.86-.514-6.67.887-1.79 2.647-2.91 4.601-3.01 1.651-.09 3.368.56 4.798 2.01 1.429-1.45 3.146-2.1 4.796-2.01 1.954.1 3.714 1.22 4.605 3.01.894 1.81.846 4.17-.514 6.67z"></path></g></svg>
          </div>
          <div id="tweet-quote-button" @click="handleQuote" class="group p-2 cursor-pointer text-[#71767B] hover:text-[#1D9BF0]">
             <svg viewBox="0 0 24 24" aria-hidden="true" class="h-6 w-6 fill-current"><g><path d="M8.75 21V3h2v18h-2zM18 21V8.5h2V21h-2zM4 21l.004-10h2L6 21H4zm9.248 0v-7h2v7h-2z"></path></g></svg>
          </div>
       </div>

       <!-- Reply Area (Visual Placeholder) -->
       <div class="py-4 flex gap-3 text-[#71767B]">
          <div class="w-12 h-12 rounded-full bg-gray-700"></div>
          <div class="flex-1 py-3 px-4 text-xl">Post your reply</div>
          <button class="bg-[#1D9BF0] opacity-50 text-white font-bold rounded-full px-4 py-1.5 self-center">Reply</button>
       </div>
    </div>
    <div v-else class="p-8 text-center text-gray-500">
       Tweet not found.
    </div>
  </div>
</template>

<script>
import { computed } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';

export default {
  name: 'TWEET_DETAIL',
  setup() {
    const router = useRouter();
    const route = useRoute();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const tweetId = computed(() => route.params.tweet_id || signatureStore.selected_tweet_id);
    const tweet = computed(() => dataStore.getTweetById(tweetId.value));
    const author = computed(() => tweet.value ? dataStore.getUserById(tweet.value.author_id) : null);

    const handleBack = () => {
      signatureStore.setCurrentPageId('HOME_TIMELINE');
      router.push({ name: 'HOME_TIMELINE' });
    };

    const handleOpenProfile = (userId) => {
      if (!userId) return;
      signatureStore.user_id = userId; // FSM uses parameter mapping, store update for simplicity here
      signatureStore.setCurrentPageId('PROFILE_OVERVIEW'); // Wait, FSM says PROFILE_OVERVIEW for user? 
      // Check FSM: ACT_TWEET_DETAIL_OPEN_AUTHOR_PROFILE -> to: PROFILE_OVERVIEW
      // Wait, PROFILE_OVERVIEW is usually "My Profile". 
      // FSM check: page PROFILE_OVERVIEW has signature profile_user_id. 
      // But page USER_PROFILE_OVERVIEW has user_id. 
      // Let's re-read FSM ACT_TWEET_DETAIL_OPEN_AUTHOR_PROFILE.
      // It goes to "PROFILE_OVERVIEW". Parameter "user_id". 
      // This seems to imply PROFILE_OVERVIEW handles both or FSM meant USER_PROFILE_OVERVIEW.
      // However, strict adherence to FSM: destination is PROFILE_OVERVIEW.
      // Wait, let's look at PROFILE_OVERVIEW signature: "profile_user_id".
      // Let's map user_id param to profile_user_id in store if we go there.
      
      // ACT_TWEET_DETAIL_OPEN_AUTHOR_PROFILE
      // from: TWEET_DETAIL, to: PROFILE_OVERVIEW
      // param: user_id
      
      // Ideally, if it's another user, it should be USER_PROFILE_OVERVIEW.
      // But I must follow FSM. If FSM says PROFILE_OVERVIEW, I go there.
      // BUT, let's check if PROFILE_OVERVIEW logic supports viewing others.
      // PROFILE_OVERVIEW actions: ACT_PROFILE_OVERVIEW_OPEN_EDIT_PROFILE.
      // This implies it's "My Profile".
      // If FSM sends me to PROFILE_OVERVIEW with a user_id, maybe it means "View Profile as Viewer"?
      // OR maybe the FSM has a "USER_PROFILE_OVERVIEW" page (id: 1546) and ACT_TWEET_DETAIL_OPEN_AUTHOR_PROFILE (id: 629) points to PROFILE_OVERVIEW (id: 965).
      // This might be a quirk. I will follow FSM but also check if I should use USER_PROFILE_OVERVIEW for logic if user != me.
      // Actually, if I look at FSM line 633: "to": "PROFILE_OVERVIEW".
      // Okay, I will route to PROFILE_OVERVIEW.
      // I'll update store `profile_user_id` with the passed `user_id`.
      
      signatureStore.profile_user_id = userId;
      signatureStore.setCurrentPageId('PROFILE_OVERVIEW');
      router.push({ name: 'PROFILE_OVERVIEW' });
    };

    const handleReply = () => {
       signatureStore.setCurrentPageId('COMPOSE_TWEET');
       router.push({ name: 'COMPOSE_TWEET' });
    };

    const handleQuote = () => {
       signatureStore.setCurrentPageId('COMPOSE_TWEET');
       router.push({ name: 'COMPOSE_TWEET' });
    };

    return {
       tweet,
       author,
       handleBack,
       handleOpenProfile,
       handleReply,
       handleQuote
    };
  }
}
</script>