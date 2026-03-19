<template>
  <div class="flex flex-col min-h-screen bg-black text-white pb-20 sm:pb-0">
    <!-- Header -->
    <div class="sticky top-0 z-30 bg-black/80 backdrop-blur-md px-4 py-3 flex items-center gap-4 border-b border-[#2F3336]">
      <div id="profile-back-home" @click="handleBackHome" class="p-2 -ml-2 rounded-full hover:bg-white/10 cursor-pointer transition-colors">
         <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 fill-current"><g><path d="M7.414 13l5.043 5.04-1.414 1.42L3.586 12l7.457-7.46 1.414 1.42L7.414 11H21v2H7.414z"></path></g></svg>
      </div>
      <div>
         <h2 class="text-xl font-bold">{{ user?.name }}</h2>
         <div class="text-sm text-[#71767B]">125 posts</div>
      </div>
    </div>

    <!-- Banner -->
    <div class="h-48 bg-[#333639] relative">
       <!-- Mock banner -->
       <img src="/images/ProfileBanner.jpg" alt="banner" class="w-full h-full object-cover opacity-50">
    </div>

    <!-- Profile Info -->
    <div class="px-4 pb-4 relative">
       <div class="flex justify-between items-start">
          <div class="w-32 h-32 rounded-full border-4 border-black -mt-16 overflow-hidden bg-gray-700 relative z-10">
             <img :src="user?.avatar || '/images/photo1766328609.jpg'" alt="avatar" class="w-full h-full object-cover">
          </div>
          <div class="mt-3">
             <button id="profile-edit-button" @click="handleEditProfile" class="border border-[#536471] text-white font-bold rounded-full px-4 py-1.5 hover:bg-white/10 transition-colors">
               Edit profile
             </button>
          </div>
       </div>

       <div class="mt-4">
          <h1 class="text-xl font-bold leading-5">{{ user?.name }}</h1>
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
          <div id="profile-following-link" @click="handleFollowing" class="hover:underline cursor-pointer">
             <span class="font-bold text-white">{{ user?.following_count }}</span>
             <span class="text-[#71767B] ml-1">Following</span>
          </div>
          <div class="hover:underline cursor-pointer">
             <span class="font-bold text-white">{{ user?.followers_count }}</span>
             <span class="text-[#71767B] ml-1">Followers</span>
          </div>
       </div>
    </div>

    <!-- Tabs -->
    <div class="flex border-b border-[#2F3336] mt-2">
       <div id="profile-tab-tweets" @click="handleTabTweets" class="flex-1 hover:bg-white/10 transition-colors cursor-pointer py-4 text-center font-bold relative">
          <span>Posts</span>
          <div class="absolute bottom-0 left-1/2 -translate-x-1/2 h-1 w-14 bg-[#1D9BF0] rounded-full"></div>
       </div>
       <div class="flex-1 hover:bg-white/10 transition-colors cursor-pointer py-4 text-center font-medium text-[#71767B]">
          Replies
       </div>
       <div class="flex-1 hover:bg-white/10 transition-colors cursor-pointer py-4 text-center font-medium text-[#71767B]">
          Media
       </div>
       <div class="flex-1 hover:bg-white/10 transition-colors cursor-pointer py-4 text-center font-medium text-[#71767B]">
          Likes
       </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';

export default {
  name: 'PROFILE_OVERVIEW',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const userId = computed(() => signatureStore.profile_user_id || 'user_me');
    const user = computed(() => dataStore.getUserById(userId.value));

    const handleBackHome = () => {
       signatureStore.setCurrentPageId('HOME');
       router.push({ name: 'HOME' });
    };

    const handleEditProfile = () => {
       signatureStore.setCurrentPageId('SETTINGS_PROFILE_EDIT');
       router.push({ name: 'SETTINGS_PROFILE_EDIT' });
    };

    const handleFollowing = () => {
       signatureStore.setCurrentPageId('PROFILE_FOLLOWING_LIST');
       router.push({ name: 'PROFILE_FOLLOWING_LIST' });
    };

    const handleTabTweets = () => {
       signatureStore.setCurrentPageId('PROFILE_TWEETS');
       router.push({ name: 'PROFILE_TWEETS' });
    };

    return {
       user,
       handleBackHome,
       handleEditProfile,
       handleFollowing,
       handleTabTweets
    };
  }
}
</script>