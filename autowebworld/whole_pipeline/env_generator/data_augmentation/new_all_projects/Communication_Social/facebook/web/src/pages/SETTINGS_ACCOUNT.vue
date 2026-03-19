<template>
  <div class="min-h-screen bg-gray-100 flex items-center justify-center p-4">
    <div class="bg-white rounded-xl shadow-lg w-full max-w-lg overflow-hidden flex flex-col">
      <!-- Header -->
      <div class="relative px-4 py-3 border-b border-gray-100 flex items-center justify-center bg-white z-10">
        <h2 class="text-lg font-bold text-gray-900">Account Settings</h2>
        <div 
          id="settings-back-home" 
          @click="goBack"
          class="absolute right-4 top-1/2 transform -translate-y-1/2 p-1 text-gray-500 hover:bg-gray-100 rounded-full cursor-pointer transition-colors"
        >
          Close
        </div>
      </div>

      <!-- Form -->
      <div class="flex-1 p-6 space-y-6">
        <div>
           <label class="block text-sm font-medium text-gray-700 mb-1">Display Name</label>
           <input 
             id="settings-name-input"
             type="text" 
             v-model="displayName"
             @input="handleNameInput"
             placeholder="Your Name"
             class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
           />
           <p class="mt-1 text-xs text-gray-500">This name will be visible to everyone.</p>
        </div>
        
        <div>
           <label class="block text-sm font-medium text-gray-700 mb-1">Default Privacy</label>
           <div class="relative">
             <button 
               id="settings-privacy-dropdown"
               @click="togglePrivacy"
               class="w-full flex items-center justify-between px-4 py-2 border border-gray-300 rounded-lg bg-white focus:outline-none hover:bg-gray-50"
             >
               <span>{{ privacyLabel || 'Select Privacy' }}</span>
               <svg class="h-4 w-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                 <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
               </svg>
             </button>
             
             <div v-if="privacyOpen" class="absolute top-full left-0 mt-1 w-full bg-white rounded-lg shadow-xl py-1 z-50 ring-1 ring-black ring-opacity-5">
               <div 
                 id="privacy-option-public"
                 @click="selectPrivacy('public')"
                 class="flex items-center gap-3 px-4 py-2 hover:bg-gray-100 cursor-pointer"
               >
                 <span>🌍</span> Public
               </div>
               <div 
                 id="privacy-option-friends"
                 @click="selectPrivacy('friends')"
                 class="flex items-center gap-3 px-4 py-2 hover:bg-gray-100 cursor-pointer"
               >
                 <span>👥</span> Friends
               </div>
               <div 
                 id="privacy-option-only-me"
                 @click="selectPrivacy('only_me')"
                 class="flex items-center gap-3 px-4 py-2 hover:bg-gray-100 cursor-pointer"
               >
                 <span>🔒</span> Only Me
               </div>
             </div>
           </div>
        </div>
      </div>

      <!-- Footer -->
      <div class="p-4 border-t border-gray-100 bg-gray-50">
        <button 
          id="settings-next-review"
          @click="goToReview"
          :disabled="!canProceed"
          class="w-full py-2 bg-blue-600 text-white font-semibold rounded-lg shadow-sm hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
        >
          Review Changes
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'SETTINGS_ACCOUNT',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    
    // Initialize
    const displayName = ref('Alex Johnson'); // Default mock
    const privacy = ref(signatureStore.privacy_option || '');
    const privacyOpen = ref(false);
    
    const privacyLabel = computed(() => {
        switch(privacy.value) {
            case 'public': return 'Public';
            case 'friends': return 'Friends';
            case 'only_me': return 'Only Me';
            default: return '';
        }
    });
    
    // Check FSM precondition manually in UI state logic if needed, 
    // but here we align with FSM that action is clickable if conditions met
    const canProceed = computed(() => {
      return signatureStore.name_input_filled && privacy.value.length > 0;
    });

    const handleNameInput = () => {
      signatureStore.name_input_filled = true; // FSM Effect
    };

    const togglePrivacy = () => {
      privacyOpen.value = !privacyOpen.value;
    };

    const selectPrivacy = (value) => {
      privacy.value = value;
      signatureStore.privacy_option = value; // FSM Effect
      privacyOpen.value = false;
    };

    const goToReview = async () => {
      if (canProceed.value) {
        signatureStore.currentPageId = 'SETTINGS_ACCOUNT_REVIEW';
        await router.push({ name: 'SETTINGS_ACCOUNT_REVIEW' });
      }
    };

    const goBack = async () => {
      // Clear
      signatureStore.name_input_filled = null;
      signatureStore.privacy_option = null;
      signatureStore.currentPageId = 'HOME';
      await router.push({ name: 'HOME' });
    };

    return {
      displayName,
      privacy,
      privacyOpen,
      privacyLabel,
      canProceed,
      handleNameInput,
      togglePrivacy,
      selectPrivacy,
      goToReview,
      goBack
    };
  }
}
</script>