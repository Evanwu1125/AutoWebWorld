<template>
  <div class="min-h-screen bg-white">
    <nav class="border-b border-gray-200">
       <div class="max-w-5xl mx-auto px-4 h-16 flex items-center justify-between">
          <button id="profile-back-home" @click="handleBackHome" class="font-serif text-2xl font-bold">Medium</button>
          <div class="flex gap-6 text-sm font-sans text-gray-500">
             <!-- FSM Actions -->
             <button id="profile-stories-link" @click="handleOpenStories" class="hover:text-black">Stories</button>
             <button id="profile-settings-link" @click="handleOpenSettings" class="hover:text-black">Settings</button>
             <button id="profile-edit-button" @click="handleEditProfile" class="hover:text-black text-green-600 font-medium">Edit Profile</button>
          </div>
       </div>
    </nav>

    <div class="max-w-5xl mx-auto px-4 py-12 flex flex-col md:flex-row gap-12">
       <!-- Left: Posts List (Simplified for overview) -->
       <div class="flex-1">
          <h2 class="text-4xl font-serif font-bold mb-12">{{ currentUser.name }}</h2>
          
          <div class="border-b border-gray-200 mb-8">
             <div class="flex gap-8">
                <div class="pb-4 border-b-2 border-black text-sm font-sans font-medium">Home</div>
                <div class="pb-4 text-gray-500 text-sm font-sans cursor-pointer hover:text-black">About</div>
             </div>
          </div>

          <!-- User's recent activity mock -->
          <div v-for="i in 3" :key="i" class="mb-10 pb-10 border-b border-gray-100 last:border-0">
             <div class="text-xs text-gray-500 font-sans mb-2 uppercase tracking-wide">Just published</div>
             <h3 class="text-xl font-bold font-serif mb-2">Sample Story Title {{ i }}</h3>
             <p class="text-gray-500 font-serif mb-3">A brief excerpt from the user's story showing their writing style...</p>
             <div class="flex items-center justify-between text-xs text-gray-400 font-sans">
                <span>Oct {{ 20 - i }} · {{ 3 + i }} min read</span>
             </div>
          </div>
       </div>

       <!-- Right: Profile Sidebar -->
       <div class="w-full md:w-80">
          <div class="sticky top-8">
             <img :src="currentUser.avatar" class="w-32 h-32 rounded-full mb-6 object-cover" />
             <div class="font-bold font-sans text-lg mb-1">{{ currentUser.name }}</div>
             <div class="text-gray-500 font-sans text-sm mb-4">@{{ currentUser.id }}</div>
             <p class="text-gray-600 font-serif text-sm mb-6">{{ currentUser.bio }}</p>
             <div class="text-green-700 font-sans text-sm mb-6 cursor-pointer hover:underline">Edit profile</div>
             
             <div class="text-gray-500 font-sans text-xs uppercase tracking-wide mb-2">Following</div>
             <div class="flex -space-x-2 overflow-hidden mb-8">
                <img v-for="j in 5" :key="j" class="inline-block h-8 w-8 rounded-full ring-2 ring-white bg-gray-200" :src="`/images/user-${j+1}.jpg`" alt=""/>
             </div>
          </div>
       </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'PROFILE_OVERVIEW',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()
    
    const currentUser = computed(() => dataStore.getUserById(signatureStore.current_user_id))

    const handleEditProfile = async () => {
       signatureStore.setCurrentPageId('PROFILE_EDIT')
       await router.push({ name: 'PROFILE_EDIT' })
    }

    const handleOpenStories = async () => {
       signatureStore.setCurrentPageId('STORIES_DRAFTS')
       await router.push({ name: 'STORIES_DRAFTS' })
    }

    const handleOpenSettings = async () => {
       signatureStore.setCurrentPageId('SETTINGS_PREFERENCES')
       await router.push({ name: 'SETTINGS_PREFERENCES' })
    }

    const handleBackHome = async () => {
       signatureStore.setCurrentPageId('HOME')
       await router.push({ name: 'HOME' })
    }

    return {
       currentUser,
       handleEditProfile,
       handleOpenStories,
       handleOpenSettings,
       handleBackHome
    }
  }
}
</script>