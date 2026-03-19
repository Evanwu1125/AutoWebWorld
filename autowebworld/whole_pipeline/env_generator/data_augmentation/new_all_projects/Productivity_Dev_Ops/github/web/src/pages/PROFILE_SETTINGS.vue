<template>
  <div class="min-h-screen bg-[#0d1117] text-white font-sans flex flex-col items-center py-12">
    <div class="w-full max-w-3xl px-6 grid grid-cols-1 md:grid-cols-4 gap-8">
        <!-- Sidebar -->
        <div class="space-y-2">
            <div id="profile-settings-back" @click="goBack" class="cursor-pointer flex items-center gap-2 text-gray-400 hover:text-blue-400 mb-4">
                <svg height="16" aria-hidden="true" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-arrow-left fill-current"><path d="M7.78 12.53a.75.75 0 0 1-1.06 0L2.47 8.28a.75.75 0 0 1 0-1.06l4.25-4.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042L4.81 7h7.44a.75.75 0 0 1 0 1.5H4.81l2.97 2.97a.75.75 0 0 1 0 1.06Z"></path></svg>
                Profile
            </div>
            <div class="font-semibold px-3 py-2 bg-[#21262d] border-l-2 border-[#fd8c73] rounded-r-md cursor-pointer">Public profile</div>
            <div class="px-3 py-2 text-gray-400 hover:bg-[#161b22] hover:text-white rounded-md cursor-pointer">Account</div>
            <div class="px-3 py-2 text-gray-400 hover:bg-[#161b22] hover:text-white rounded-md cursor-pointer">Appearance</div>
            <div class="px-3 py-2 text-gray-400 hover:bg-[#161b22] hover:text-white rounded-md cursor-pointer">Notifications</div>
        </div>

        <!-- Form -->
        <div class="col-span-3">
             <h1 class="text-2xl font-normal mb-6 pb-2 border-b border-gray-700">Public profile</h1>
             
             <div class="space-y-6">
                 <div>
                     <label class="block font-semibold mb-2">Name</label>
                     <input 
                       id="profile-name-input"
                       type="text" 
                       v-model="profileName"
                       class="w-full px-3 py-2 bg-[#0d1117] border border-gray-600 rounded-md focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none"
                     />
                     <p class="text-xs text-gray-500 mt-1">Your name may appear around GitHub where you contribute or are mentioned.</p>
                 </div>

                 <div>
                     <label class="block font-semibold mb-2">Public email</label>
                     <select class="w-full px-3 py-2 bg-[#0d1117] border border-gray-600 rounded-md focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none appearance-none">
                         <option>Select a verified email to display</option>
                         <option>octocat@github.com</option>
                     </select>
                 </div>

                 <div>
                     <label class="block font-semibold mb-2">Bio</label>
                     <textarea 
                       id="profile-bio-input"
                       v-model="profileBio"
                       class="w-full bg-[#0d1117] border border-gray-600 rounded-md p-3 min-h-[100px] focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none"
                       placeholder="Tell us a little bit about yourself"
                     ></textarea>
                 </div>

                 <div>
                     <label class="block font-semibold mb-2">Location</label>
                     <input 
                       id="profile-location-input"
                       type="text" 
                       v-model="profileLocation"
                       class="w-full px-3 py-2 bg-[#0d1117] border border-gray-600 rounded-md focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none"
                     />
                 </div>

                 <div>
                     <label class="block font-semibold mb-2">Website</label>
                     <input 
                       id="profile-website-input"
                       type="text" 
                       v-model="profileWebsite"
                       class="w-full px-3 py-2 bg-[#0d1117] border border-gray-600 rounded-md focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none"
                     />
                 </div>
                 
                 <div class="pt-4">
                     <button 
                       id="profile-save-button" 
                       @click="saveProfile"
                       :disabled="!profileName"
                       :class="['px-4 py-2 text-white font-medium rounded-md', profileName ? 'bg-[#238636] hover:bg-[#2ea44f]' : 'bg-gray-700 cursor-not-allowed opacity-50']"
                     >
                         Update profile
                     </button>
                 </div>
             </div>
        </div>
    </div>
  </div>
</template>

<script>
import { ref, watch } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { FSMRuntime } from '../fsm/FSMRuntime';
import fsmData from '../../fsm.json';

export default {
  name: 'PROFILE_SETTINGS',
  setup() {
    const router = useRouter();
    const store = useSignatureStore();
    const fsmRuntime = new FSMRuntime(fsmData, { store, router });

    const profileName = ref('The Octocat');
    const profileBio = ref('GitHub Mascot');
    const profileLocation = ref('San Francisco');
    const profileWebsite = ref('https://github.com');

    // Sync
    watch(profileName, val => store.signature.profile_name = val);
    watch(profileBio, val => store.signature.profile_bio = val);
    watch(profileLocation, val => store.signature.profile_location = val);
    watch(profileWebsite, val => store.signature.profile_website = val);

    // Init
    store.signature.profile_name = 'The Octocat';
    
    const saveProfile = async () => {
        const action = fsmData.pages.find(p => p.id === 'PROFILE_SETTINGS').actions.find(a => a.id === 'ACT_PROFILE_SETTINGS_SAVE');
        if (action && fsmRuntime.checkPreconditions(action, store.signature)) {
            store.setCurrentPageId('PROFILE_UPDATE_SUCCESS');
            await router.push({ name: 'PROFILE_UPDATE_SUCCESS' });
        }
    };

    const goBack = async () => {
        store.setCurrentPageId('PROFILE_OVERVIEW');
        await router.push({ name: 'PROFILE_OVERVIEW' });
    };

    return {
        profileName,
        profileBio,
        profileLocation,
        profileWebsite,
        saveProfile,
        goBack
    };
  }
}
</script>