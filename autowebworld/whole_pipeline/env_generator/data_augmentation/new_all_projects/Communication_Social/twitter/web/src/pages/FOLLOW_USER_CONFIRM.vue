<template>
  <div class="flex flex-col items-center justify-center min-h-screen bg-black text-white p-6 relative">
    <!-- Header Back Button (Absolute) -->
    <div class="absolute top-4 left-4 z-30">
        <div id="confirm-follow-back" @click="handleBack" class="p-2 rounded-full hover:bg-white/10 cursor-pointer transition-colors bg-black/50">
             <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 fill-current"><g><path d="M7.414 13l5.043 5.04-1.414 1.42L3.586 12l7.457-7.46 1.414 1.42L7.414 11H21v2H7.414z"></path></g></svg>
        </div>
    </div>

    <div class="bg-black border border-[#2F3336] rounded-2xl p-8 max-w-sm w-full text-center">
       <div class="w-16 h-16 rounded-full overflow-hidden bg-gray-700 mx-auto mb-4 border-2 border-black">
           <img :src="targetUser?.avatar || '/images/photo1766328669.jpg'" alt="avatar" class="w-full h-full object-cover">
       </div>
       
       <h2 class="text-xl font-bold mb-1">Follow {{ targetUser?.name }}?</h2>
       <div class="text-[#71767B] mb-6">{{ targetUser?.handle }}</div>
       
       <p class="text-sm text-[#71767B] mb-6">
           You will see their Tweets in your Home Timeline. You can unfollow them at any time.
       </p>

       <div class="flex items-center justify-center gap-2 mb-6 cursor-pointer" @click="toggleCheckbox">
          <input 
             id="confirm-follow-checkbox" 
             type="checkbox" 
             v-model="isChecked"
             class="form-checkbox bg-transparent border-[#536471] text-[#1D9BF0] rounded focus:ring-0 focus:ring-offset-0 h-5 w-5"
          >
          <label class="text-sm cursor-pointer select-none">I confirm I want to follow</label>
       </div>

       <button 
          id="confirm-follow-submit" 
          @click="handleSubmit" 
          :disabled="!isChecked"
          :class="isChecked ? 'bg-white hover:bg-[#EFF3F4] text-black' : 'bg-[#787a7a] cursor-not-allowed text-[#16181C]'"
          class="w-full font-bold rounded-full py-3 transition-colors"
       >
          Follow
       </button>
    </div>
  </div>
</template>

<script>
import { ref, computed, watch } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';

export default {
  name: 'FOLLOW_USER_CONFIRM',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const targetUserId = computed(() => signatureStore.target_user_id);
    const targetUser = computed(() => dataStore.getUserById(targetUserId.value));
    
    const isChecked = ref(false);

    const toggleCheckbox = () => {
        // Handled by v-model mostly, but if clicking label wrapper:
        // isChecked.value = !isChecked.value; 
        // We let input handle it via click
    };
    
    watch(isChecked, (val) => {
        if (val) signatureStore.confirm_checked = true;
        else signatureStore.confirm_checked = null; 
    });

    const handleSubmit = () => {
        if (!isChecked.value) return;
        
        signatureStore.setCurrentPageId('FOLLOW_USER_SUCCESS');
        router.push({ name: 'FOLLOW_USER_SUCCESS' });
    };

    const handleBack = () => {
        signatureStore.setCurrentPageId('USER_PROFILE_OVERVIEW');
        router.push({ name: 'USER_PROFILE_OVERVIEW', params: { user_id: targetUserId.value } });
    };

    return {
        targetUser,
        isChecked,
        toggleCheckbox,
        handleSubmit,
        handleBack
    };
  }
}
</script>