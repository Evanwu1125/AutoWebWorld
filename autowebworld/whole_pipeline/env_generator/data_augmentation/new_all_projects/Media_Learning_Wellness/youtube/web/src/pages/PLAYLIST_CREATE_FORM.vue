<template>
  <div class="min-h-screen bg-[#0F0F0F] text-white flex flex-col items-center justify-center p-6 relative">
    <!-- Overlay Background -->
    <div class="absolute inset-0 bg-[url('/images/Background.jpg')] bg-cover opacity-10 blur-sm"></div>

    <div class="bg-[#1F1F1F] rounded-2xl p-8 max-w-lg w-full shadow-2xl border border-gray-800 relative z-10">
      <h2 class="text-2xl font-bold mb-6">Create New Playlist</h2>
      
      <!-- Title Input -->
      <div class="mb-6">
        <label class="block text-sm font-medium text-gray-400 mb-2">Title</label>
        <div class="relative group">
          <input 
            id="playlist-title-input"
            v-model="title"
            @input="handleTitleInput"
            type="text"
            placeholder="Enter playlist title"
            class="w-full bg-[#121212] border border-gray-700 rounded-lg px-4 py-3 focus:border-blue-500 focus:outline-none transition-colors"
          >
          <div class="absolute bottom-0 left-0 w-0 h-0.5 bg-blue-500 transition-all group-focus-within:w-full"></div>
        </div>
      </div>

      <!-- Description Input -->
      <div class="mb-6">
        <label class="block text-sm font-medium text-gray-400 mb-2">Description</label>
        <textarea 
          id="playlist-description-input"
          v-model="description"
          @input="handleDescriptionInput"
          placeholder="Tell viewers about your playlist"
          rows="3"
          class="w-full bg-[#121212] border border-gray-700 rounded-lg px-4 py-3 focus:border-blue-500 focus:outline-none transition-colors resize-none"
        ></textarea>
      </div>

      <!-- Privacy Dropdown -->
      <div class="mb-8 relative">
        <label class="block text-sm font-medium text-gray-400 mb-2">Privacy</label>
        <button 
          id="playlist-privacy-dropdown"
          @click="isPrivacyOpen = !isPrivacyOpen"
          class="w-full bg-[#121212] border border-gray-700 rounded-lg px-4 py-3 text-left flex items-center justify-between hover:border-gray-500 transition-colors"
        >
          <span class="capitalize">{{ privacy || 'Select Privacy' }}</span>
          <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
        </button>

        <div v-if="isPrivacyOpen" class="absolute w-full mt-2 bg-[#272727] border border-gray-700 rounded-xl shadow-xl z-20 overflow-hidden">
          <div 
            id="playlist-privacy-option-public" 
            @click="selectPrivacy('public')"
            class="px-4 py-3 hover:bg-gray-700 cursor-pointer flex items-center gap-3"
          >
            <svg class="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
            <div>
              <div class="font-medium">Public</div>
              <div class="text-xs text-gray-400">Anyone can search for and view</div>
            </div>
          </div>
          <div 
            id="playlist-privacy-option-unlisted" 
            @click="selectPrivacy('unlisted')"
            class="px-4 py-3 hover:bg-gray-700 cursor-pointer flex items-center gap-3"
          >
            <svg class="w-5 h-5 text-yellow-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"></path></svg>
            <div>
              <div class="font-medium">Unlisted</div>
              <div class="text-xs text-gray-400">Anyone with the link can view</div>
            </div>
          </div>
          <div 
            id="playlist-privacy-option-private" 
            @click="selectPrivacy('private')"
            class="px-4 py-3 hover:bg-gray-700 cursor-pointer flex items-center gap-3"
          >
            <svg class="w-5 h-5 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"></path></svg>
            <div>
              <div class="font-medium">Private</div>
              <div class="text-xs text-gray-400">Only you can view</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Actions -->
      <div class="flex gap-4 pt-4 border-t border-gray-800">
        <button 
          id="playlist-create-cancel" 
          @click="goBackLibrary"
          class="flex-1 py-3 rounded-full font-medium hover:bg-white/10 transition-colors"
        >
          Cancel
        </button>
        <button 
          id="playlist-create-submit" 
          @click="submitForm"
          class="flex-1 bg-[#3EA6FF] text-black font-bold py-3 rounded-full hover:bg-blue-400 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          :disabled="!isValid"
        >
          Create
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'PLAYLIST_CREATE_FORM',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const title = ref('')
    const description = ref('')
    const privacy = ref(null)
    const isPrivacyOpen = ref(false)

    const isValid = computed(() => {
      return title.value.length > 0 && privacy.value !== null
    })

    const handleTitleInput = () => {
      if (title.value.length > 0) store.playlist_title_entered = 'typed'
      else store.playlist_title_entered = null
    }

    const handleDescriptionInput = () => {
      if (description.value.length > 0) store.playlist_description_entered = 'typed'
    }

    const selectPrivacy = (val) => {
      privacy.value = val
      store.playlist_privacy_selected = 'public' // FSM simplifies to checking > 0 length, but specifically sets 'public' in effects. We set any value to trigger validation.
      isPrivacyOpen.value = false
    }

    const submitForm = () => {
      if (isValid.value) {
        store.currentPageId = 'PLAYLIST_CREATE_SUCCESS'
        router.push({ name: 'PLAYLIST_CREATE_SUCCESS' })
      }
    }

    const goBackLibrary = () => {
      store.currentPageId = 'LIBRARY'
      router.push({ name: 'LIBRARY' })
    }

    return {
      title,
      description,
      privacy,
      isPrivacyOpen,
      isValid,
      handleTitleInput,
      handleDescriptionInput,
      selectPrivacy,
      submitForm,
      goBackLibrary
    }
  }
}
</script>