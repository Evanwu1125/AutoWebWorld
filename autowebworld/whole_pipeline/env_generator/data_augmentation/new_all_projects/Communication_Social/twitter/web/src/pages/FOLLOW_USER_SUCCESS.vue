<template>
  <div class="flex flex-col items-center justify-center min-h-screen bg-black text-white p-6 text-center">
    <div class="w-16 h-16 bg-[#00BA7C] rounded-full flex items-center justify-center mb-6">
       <svg viewBox="0 0 24 24" aria-hidden="true" class="h-8 w-8 fill-white"><g><path d="M9 16.17l-4.17-4.17-1.42 1.42L9 19 21 7l-1.41-1.41z"></path></g></svg>
    </div>
    
    <h2 class="text-2xl font-bold mb-2">Following!</h2>
    <p class="text-[#71767B] mb-8">You are now following {{ targetUser?.name }}</p>
    
    <div class="flex flex-col gap-4 w-full max-w-xs">
       <button id="follow-success-view-profile" @click="handleViewProfile" class="w-full border border-[#536471] text-white font-bold rounded-full py-3 hover:bg-white/10 transition-colors">
          View Profile
       </button>
       <button id="follow-success-go-home" @click="handleGoHome" class="w-full bg-white text-black font-bold rounded-full py-3 hover:bg-[#EFF3F4] transition-colors">
          Back to Home
       </button>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';

export default {
  name: 'FOLLOW_USER_SUCCESS',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const targetUserId = computed(() => signatureStore.target_user_id);
    const targetUser = computed(() => dataStore.getUserById(targetUserId.value));

    const handleViewProfile = () => {
       signatureStore.user_id = targetUserId.value;
       signatureStore.setCurrentPageId('USER_PROFILE_OVERVIEW');
       router.push({ name: 'USER_PROFILE_OVERVIEW', params: { user_id: targetUserId.value } });
    };

    const handleGoHome = () => {
       signatureStore.setCurrentPageId('HOME');
       router.push({ name: 'HOME' });
    };

    return {
       targetUser,
       handleViewProfile,
       handleGoHome
    };
  }
}
</script>