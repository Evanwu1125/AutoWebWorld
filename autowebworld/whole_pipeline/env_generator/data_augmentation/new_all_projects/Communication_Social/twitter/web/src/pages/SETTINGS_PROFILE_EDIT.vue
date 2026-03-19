<template>
  <div class="flex flex-col min-h-screen bg-black text-white p-6 relative">
    <!-- Header -->
    <div class="sticky top-0 z-30 bg-black/80 backdrop-blur-md flex items-center justify-between mb-6">
      <div class="flex items-center gap-4">
        <div id="settings-back-profile" @click="handleBack" class="p-2 -ml-2 rounded-full hover:bg-white/10 cursor-pointer transition-colors">
            <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 fill-current"><g><path d="M10.59 12L4.54 5.96l1.42-1.42L12 10.59l6.04-6.05 1.42 1.42L13.41 12l6.05 6.04-1.42 1.42L12 13.41l-6.04 6.05-1.42-1.42L10.59 12z"></path></g></svg>
        </div>
        <h2 class="text-xl font-bold">Edit Profile</h2>
      </div>
      <button 
         id="settings-save-button" 
         @click="handleSave"
         :disabled="!isValid"
         :class="isValid ? 'bg-white text-black hover:bg-[#EFF3F4]' : 'bg-[#787a7a] text-[#16181C] cursor-not-allowed'"
         class="font-bold rounded-full px-4 py-1.5 transition-colors"
      >
        Save
      </button>
    </div>

    <!-- Cover & Avatar (Visual) -->
    <div class="h-32 bg-[#333639] relative mb-12 opacity-50">
       <div class="absolute -bottom-10 left-4 w-20 h-20 rounded-full bg-gray-700 border-4 border-black overflow-hidden">
          <img src="/images/photo1766328835.jpg" alt="avatar" class="w-full h-full object-cover">
       </div>
    </div>

    <!-- Form -->
    <div class="flex flex-col gap-6">
       <!-- Name -->
       <div class="border border-[#333639] rounded px-2 py-1 focus-within:border-[#1D9BF0] focus-within:ring-1 focus-within:ring-[#1D9BF0] transition-colors">
          <label class="text-xs text-[#71767B] block">Name</label>
          <input 
             id="settings-display-name-input"
             v-model="displayName"
             @input="handleNameInput"
             type="text" 
             class="w-full bg-transparent text-white focus:outline-none py-1"
          >
       </div>

       <!-- Bio -->
       <div class="border border-[#333639] rounded px-2 py-1 focus-within:border-[#1D9BF0] focus-within:ring-1 focus-within:ring-[#1D9BF0] transition-colors">
          <label class="text-xs text-[#71767B] block">Bio</label>
          <textarea 
             id="settings-bio-textarea"
             v-model="bio"
             @input="handleBioInput"
             class="w-full bg-transparent text-white focus:outline-none py-1 resize-none h-24"
          ></textarea>
       </div>

       <!-- Location -->
       <div class="border border-[#333639] rounded px-2 py-1 focus-within:border-[#1D9BF0] focus-within:ring-1 focus-within:ring-[#1D9BF0] transition-colors">
          <label class="text-xs text-[#71767B] block">Location</label>
          <input 
             id="settings-location-input"
             v-model="location"
             @input="handleLocationInput"
             type="text" 
             class="w-full bg-transparent text-white focus:outline-none py-1"
          >
       </div>
       
       <!-- Website (Optional, not in FSM actions but standard UI) -->
       <div class="border border-[#333639] rounded px-2 py-1 focus-within:border-[#1D9BF0] focus-within:ring-1 focus-within:ring-[#1D9BF0] transition-colors opacity-50">
          <label class="text-xs text-[#71767B] block">Website</label>
          <input type="text" placeholder="Add your website" class="w-full bg-transparent text-white focus:outline-none py-1" disabled>
       </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';

export default {
  name: 'SETTINGS_PROFILE_EDIT',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();
    
    // Init with current user data
    const currentUser = dataStore.getUserById('user_me');

    const displayName = ref(currentUser?.name || '');
    const bio = ref(currentUser?.bio || '');
    const location = ref(currentUser?.location || '');

    // Preconditions: name > 0, bio > 0
    const isValid = computed(() => displayName.value.length > 0 && bio.value.length > 0);

    const handleNameInput = () => {
        signatureStore.display_name = displayName.value;
    };

    const handleBioInput = () => {
        signatureStore.bio = bio.value;
    };

    const handleLocationInput = () => {
        signatureStore.location = location.value;
    };

    const handleSave = () => {
        if (!isValid.value) return;
        
        // Update mock data store directly for immediate feedback (FSM effect only updates signature usually, but we want persistence)
        currentUser.name = displayName.value;
        currentUser.bio = bio.value;
        currentUser.location = location.value;
        
        signatureStore.setCurrentPageId('PROFILE_UPDATE_SUCCESS');
        router.push({ name: 'PROFILE_UPDATE_SUCCESS' });
    };

    const handleBack = () => {
        signatureStore.setCurrentPageId('PROFILE_OVERVIEW');
        router.push({ name: 'PROFILE_OVERVIEW' });
    };

    return {
        displayName,
        bio,
        location,
        isValid,
        handleNameInput,
        handleBioInput,
        handleLocationInput,
        handleSave,
        handleBack
    };
  }
}
</script>