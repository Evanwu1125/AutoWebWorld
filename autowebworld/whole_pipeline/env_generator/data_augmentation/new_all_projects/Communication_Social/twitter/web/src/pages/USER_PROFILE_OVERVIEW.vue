<template>
  <div class="flex flex-col min-h-screen bg-black text-white pb-20 sm:pb-0">
    <!-- Header -->
    <div class="sticky top-0 z-30 bg-black/80 backdrop-blur-md px-4 py-3 flex items-center gap-4 border-b border-[#2F3336]">
      <div id="user-profile-back-following" @click="handleBackFollowing" class="p-2 -ml-2 rounded-full hover:bg-white/10 cursor-pointer transition-colors">
         <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 fill-current"><g><path d="M7.414 13l5.043 5.04-1.414 1.42L3.586 12l7.457-7.46 1.414 1.42L7.414 11H21v2H7.414z"></path></g></svg>
      </div>
      <div>
         <h2 class="text-xl font-bold">{{ user?.name }}</h2>
         <div class="text-sm text-[#71767B]">500 posts</div>
      </div>
    </div>

    <!-- Banner -->
    <div class="h-48 bg-[#333639] relative">
       <img src="/images/photo1766328666.jpg" alt="banner" class="w-full h-full object-cover opacity-50">
    </div>

    <!-- Profile Info -->
    <div class="px-4 pb-4 relative">
       <div class="flex justify-between items-start">
          <div class="w-32 h-32 rounded-full border-4 border-black -mt-16 overflow-hidden bg-gray-700 relative z-10">
             <img :src="user?.avatar || '/images/photo1766328665.jpg'" alt="avatar" class="w-full h-full object-cover">
          </div>
          <div class="mt-3 flex gap-2">
             <div id="user-profile-message-button" @click="handleMessage" class="p-2 border border-[#536471] rounded-full hover:bg-white/10 cursor-pointer transition-colors">
                <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 fill-white"><g><path d="M1.998 5.5c0-1.381 1.119-2.5 2.5-2.5h15c1.381 0 2.5 1.119 2.5 2.5v13c0 1.381-1.119 2.5-2.5 2.5h-15c-1.381 0-2.5-1.119-2.5-2.5v-13zm2.5-.5c-.276 0-.5.224-.5.5v2.764l8 3.638 8-3.636V5.5c0-.276-.224-.5-.5-.5h-15zm15.5 5.463l-8 3.636-8-3.638V18.5c0 .276.224.5.5.5h15c.276 0 .5-.224.5-.5v-8.037z"></path></g></svg>
             </div>
             <button id="user-profile-follow-button" @click="handleFollow" class="bg-white text-black font-bold rounded-full px-4 py-1.5 hover:bg-[#EFF3F4] transition-colors">
               Follow
             </button>
          </div>
       </div>

       <div class="mt-4">
          <h1 class="text-xl font-bold leading-5 flex items-center gap-1">
              {{ user?.name }}
              <svg v-if="user?.verified" viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 text-[#1D9BF0] fill-current"><g><path d="M22.5 12.5c0-1.58-.875-2.95-2.148-3.6.154-.435.238-.905.238-1.4 0-2.21-1.71-3.998-3.818-3.998-.47 0-.92.084-1.336.25C14.818 2.415 13.51 1.5 12 1.5s-2.816.917-3.437 2.25c-.415-.165-.866-.25-1.336-.25-2.11 0-3.818 1.79-3.818 4 0 .495.083.965.238 1.4-1.272.65-2.147 2.018-2.147 3.6 0 1.495.782 2.798 1.942 3.486-.02.17-.032.34-.032.514 0 2.21 1.708 4 3.818 4 .47 0 .92-.086 1.335-.25.62 1.334 1.926 2.25 3.437 2.25 1.512 0 2.818-.916 3.437-2.25.415.163.865.248 1.336.248 2.11 0 3.818-1.79 3.818-4 0-.174-.012-.344-.033-.513 1.158-.687 1.943-1.99 1.943-3.484zm-6.616-3.334l-4.334 6.5c-.145.217-.382.334-.625.334-.143 0-.288-.04-.416-.126l-.115-.094-2.415-2.415c-.293-.293-.293-.768 0-1.06s.768-.294 1.06 0l1.77 1.767 3.825-5.74c.23-.345.696-.436 1.04-.207.346.23.44.696.21 1.04z"></path></g></svg>
          </h1>
          <div class="text-[#71767B] text-sm mt-1">{{ user?.handle }}</div>
       </div>

       <div class="mt-4 text-[15px] whitespace-pre-wrap">{{ user?.bio || 'No bio yet.' }}</div>

       <div class="flex flex-wrap gap-x-4 gap-y-2 mt-3 text-[#71767B] text-[15px]">
          <div v-if="user?.location" class="flex items-center gap-1">
             <svg viewBox="0 0 24 24" aria-hidden="true" class="h-4.5 w-4.5 fill-current"><g><path d="M12 7c-1.93 0-3.5 1.57-3.5 3.5S10.07 14 12 14s3.5-1.57 3.5-3.5S13.93 7 12 7zm0 9c-3.033 0-5.5-2.467-5.5-5.5S8.967 5.5 12 5.5 17.5 7.967 17.5 11 15.033 16 12 16zm0-12.5C7.306 3.5 3.5 7.306 3.5 12 3.5 17.42 12 21.25 12 21.25S20.5 17.42 20.5 12C20.5 7.306 16.694 3.5 12 3.5z"></path></g></svg>
             <span>{{ user?.location }}</span>
          </div>
          <div class="flex items-center gap-1">
             <svg viewBox="0 0 24 24" aria-hidden="true" class="h-4.5 w-4.5 fill-current"><g><path d="M7 4V3h2v1h6V3h2v1h1.5C19.89 4 21 5.12 21 6.5v12c0 1.38-1.11 2.5-2.5 2.5h-13C4.12 21 3 19.88 3 18.5v-12C3 5.12 4.12 4 5.5 4H7zm0 2H5.5c-.27 0-.5.22-.5.5v12c0 .28.23.5.5.5h13c.28 0 .5-.22.5-.5v-12c0-.28-.22-.5-.5-.5H17v1h-2V6H9v1H7V6zm0 6h2v-2H7v2zm0 4h2v-2H7v2zm4-4h2v-2h-2v2zm0 4h2v-2h-2v2zm4-4h2v-2h-2v2z"></path></g></svg>
             <span>Joined {{ user?.joined_date }}</span>
          </div>
       </div>

       <div class="flex gap-4 mt-3 text-[15px]">
          <div class="hover:underline cursor-pointer">
             <span class="font-bold text-white">{{ user?.following_count }}</span>
             <span class="text-[#71767B] ml-1">Following</span>
          </div>
          <div class="hover:underline cursor-pointer">
             <span class="font-bold text-white">{{ user?.followers_count }}</span>
             <span class="text-[#71767B] ml-1">Followers</span>
          </div>
       </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';

export default {
  name: 'USER_PROFILE_OVERVIEW',
  setup() {
    const router = useRouter();
    const route = useRoute();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const userId = computed(() => route.params.user_id || signatureStore.user_id);
    const user = computed(() => dataStore.getUserById(userId.value));

    const handleBackFollowing = () => {
       signatureStore.setCurrentPageId('PROFILE_FOLLOWING_LIST');
       router.push({ name: 'PROFILE_FOLLOWING_LIST' });
    };

    const handleMessage = () => {
       signatureStore.setCurrentPageId('MESSAGES_COMPOSE');
       router.push({ name: 'MESSAGES_COMPOSE' });
    };

    const handleFollow = () => {
       signatureStore.target_user_id = userId.value;
       signatureStore.setCurrentPageId('FOLLOW_USER_CONFIRM');
       router.push({ name: 'FOLLOW_USER_CONFIRM' });
    };

    return {
       user,
       handleBackFollowing,
       handleMessage,
       handleFollow
    };
  }
}
</script>