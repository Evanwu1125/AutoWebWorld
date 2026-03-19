<template>
  <div class="min-h-screen bg-white">
    <!-- Nav -->
    <nav class="border-b border-gray-200">
       <div class="max-w-3xl mx-auto px-4 h-16 flex items-center justify-between">
          <div class="flex items-center gap-4">
             <button id="profile-edit-back" @click="handleBack" class="p-2 hover:bg-gray-100 rounded-full transition-colors">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                </svg>
             </button>
             <span class="font-serif font-bold text-lg">Profile Information</span>
          </div>
       </div>
    </nav>

    <div class="max-w-2xl mx-auto px-4 py-12">
       <div class="flex items-start gap-8 mb-12">
          <div class="flex-1 space-y-6">
             <div>
                <label class="block text-sm font-sans font-medium text-gray-700 mb-1">Name</label>
                <input 
                   id="profile-name-input"
                   v-model="name"
                   type="text" 
                   class="w-full border-b border-gray-300 focus:border-black focus:ring-0 border-t-0 border-l-0 border-r-0 px-0 py-2 font-serif text-lg bg-transparent"
                />
             </div>
             <div>
                <label class="block text-sm font-sans font-medium text-gray-700 mb-1">Bio</label>
                <input 
                   id="profile-bio-input"
                   v-model="bio"
                   type="text" 
                   class="w-full border-b border-gray-300 focus:border-black focus:ring-0 border-t-0 border-l-0 border-r-0 px-0 py-2 font-serif text-lg bg-transparent"
                />
             </div>
             <div>
                <label class="block text-sm font-sans font-medium text-gray-700 mb-1">Location</label>
                <input 
                   id="profile-location-input"
                   v-model="location"
                   type="text" 
                   class="w-full border-b border-gray-300 focus:border-black focus:ring-0 border-t-0 border-l-0 border-r-0 px-0 py-2 font-serif text-lg bg-transparent"
                />
             </div>
          </div>
          <div class="w-32">
             <div class="text-xs text-gray-500 font-sans mb-2">Photo</div>
             <img :src="currentUser.avatar" class="w-24 h-24 rounded-full object-cover mb-2" />
             <div class="text-green-600 text-sm font-sans cursor-pointer">Update</div>
          </div>
       </div>

       <div class="border-t border-gray-200 pt-8 flex items-center justify-between">
          <div class="flex items-center gap-3">
             <input 
                type="checkbox" 
                id="profile-enable-save" 
                @click="handleEnableSave"
                :disabled="!hasChanges"
                class="rounded text-green-600 focus:ring-green-500 border-gray-300" 
             />
             <label for="profile-enable-save" class="text-sm text-gray-600 font-sans">Confirm changes</label>
          </div>

          <button 
             id="profile-save-button"
             @click="handleSave"
             :disabled="!canSave"
             :class="{
                'px-6 py-2 rounded-full text-sm font-medium font-sans transition-colors': true,
                'bg-green-600 text-white hover:bg-green-700': canSave,
                'bg-gray-200 text-gray-400 cursor-not-allowed': !canSave
             }"
          >
             Save
          </button>
       </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'PROFILE_EDIT',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()
    
    const currentUser = computed(() => dataStore.getUserById(signatureStore.current_user_id))

    const name = ref(currentUser.value.name)
    const bio = ref(currentUser.value.bio)
    const location = ref(currentUser.value.location || '')
    
    const hasChanges = computed(() => name.value !== currentUser.value.name || bio.value !== currentUser.value.bio || location.value !== currentUser.value.location)
    const canSave = computed(() => signatureStore.profile_can_save === true)

    watch(name, () => signatureStore.edit_name = 'typed')
    watch(bio, () => signatureStore.edit_bio = 'typed')
    watch(location, () => signatureStore.edit_location = 'typed')

    const handleEnableSave = () => {
       if (hasChanges.value) {
          signatureStore.profile_can_save = true
       }
    }

    const handleSave = async () => {
       if (canSave.value) {
          // Mock update data store? FSM doesn't strictly require updating the data store, but good for realism
          // But action handlers just update signature.
          signatureStore.setCurrentPageId('PROFILE_UPDATE_SUCCESS')
          await router.push({ name: 'PROFILE_UPDATE_SUCCESS' })
       }
    }

    const handleBack = async () => {
       signatureStore.setCurrentPageId('PROFILE_OVERVIEW')
       await router.push({ name: 'PROFILE_OVERVIEW' })
    }

    return {
       currentUser,
       name,
       bio,
       location,
       hasChanges,
       canSave,
       handleEnableSave,
       handleSave,
       handleBack
    }
  }
}
</script>