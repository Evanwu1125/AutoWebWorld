<template>
  <div class="min-h-screen bg-[#F1F2F2]">
    <nav class="bg-white shadow-sm sticky top-0 z-50">
      <div class="max-w-5xl mx-auto px-4 h-14 flex items-center justify-between">
        <div class="flex items-center gap-4">
          <button id="profile-back-home" @click="goHome" class="text-gray-500 hover:text-gray-700 p-2 rounded-full hover:bg-gray-100 transition-colors">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>
          </button>
          <h1 class="text-[#B92B27] text-xl font-bold font-serif">Profile</h1>
        </div>
      </div>
    </nav>

    <main class="max-w-4xl mx-auto px-4 py-8">
      <div class="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
        <!-- Cover Photo (Mock) -->
        <div class="h-48 bg-gradient-to-r from-blue-500 to-indigo-600 relative"></div>
        
        <div class="px-8 pb-8">
          <!-- Avatar & Header -->
          <div class="relative flex justify-between items-end -mt-12 mb-6">
            <div class="rounded-full p-1 bg-white shadow-lg">
               <img src="/images/photo1765097818.jpg" class="w-32 h-32 rounded-full object-cover border-4 border-white" />
            </div>
            
            <button 
              id="profile-edit-button" 
              @click="editProfile" 
              class="mb-2 bg-white text-gray-700 border border-gray-300 px-4 py-2 rounded-full font-medium hover:bg-gray-50 transition-colors shadow-sm flex items-center gap-2"
            >
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z"></path></svg>
              Edit Profile
            </button>
          </div>
          
          <!-- User Info -->
          <div class="mb-8">
            <h1 class="text-3xl font-bold text-gray-900 mb-2 font-serif">{{ profileName }}</h1>
            <p class="text-gray-600 text-lg leading-relaxed">{{ profileBio }}</p>
          </div>
          
          <!-- Stats -->
          <div class="flex gap-8 border-t border-gray-100 pt-6">
            <div class="text-center">
              <span class="block text-2xl font-bold text-gray-900">42</span>
              <span class="text-sm text-gray-500 uppercase tracking-wide">Answers</span>
            </div>
            <div class="text-center">
              <span class="block text-2xl font-bold text-gray-900">15</span>
              <span class="text-sm text-gray-500 uppercase tracking-wide">Questions</span>
            </div>
            <div class="text-center">
              <span class="block text-2xl font-bold text-gray-900">1.2k</span>
              <span class="text-sm text-gray-500 uppercase tracking-wide">Followers</span>
            </div>
            <div class="text-center">
               <span class="block text-2xl font-bold text-gray-900">350</span>
               <span class="text-sm text-gray-500 uppercase tracking-wide">Following</span>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Content Tabs (Visual Only) -->
      <div class="mt-6 flex gap-6 border-b border-gray-200">
        <button class="pb-3 border-b-2 border-[#B92B27] text-[#B92B27] font-medium">Profile</button>
        <button class="pb-3 border-b-2 border-transparent text-gray-500 hover:text-gray-700 font-medium">Answers</button>
        <button class="pb-3 border-b-2 border-transparent text-gray-500 hover:text-gray-700 font-medium">Questions</button>
        <button class="pb-3 border-b-2 border-transparent text-gray-500 hover:text-gray-700 font-medium">Followers</button>
      </div>
    </main>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'PROFILE',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const profileName = computed(() => store.profile_name)
    const profileBio = computed(() => store.profile_bio)

    function goHome() {
      store.setCurrentPageId('HOME')
      router.push({ name: 'HOME' })
    }

    function editProfile() {
      store.setCurrentPageId('PROFILE_EDIT')
      router.push({ name: 'PROFILE_EDIT' })
    }

    return {
      profileName,
      profileBio,
      goHome,
      editProfile
    }
  }
}
</script>