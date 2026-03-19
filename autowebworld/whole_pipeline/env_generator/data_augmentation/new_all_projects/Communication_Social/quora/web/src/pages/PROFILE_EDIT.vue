<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center py-10 px-4">
    <div class="bg-white w-full max-w-lg rounded-xl shadow-xl overflow-hidden">
      <!-- Header -->
      <div class="px-8 py-6 border-b border-gray-100 bg-white flex justify-between items-center">
        <h2 class="text-xl font-bold text-gray-900">Edit Profile</h2>
        <button id="profile-edit-back" @click="cancel" class="text-gray-400 hover:text-gray-600 transition-colors">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
        </button>
      </div>

      <!-- Form -->
      <div class="p-8 space-y-6">
        <div class="flex justify-center mb-6">
          <div class="relative">
             <img src="/images/photo1765097821.jpg" class="w-24 h-24 rounded-full object-cover border-4 border-gray-100" />
             <button class="absolute bottom-0 right-0 bg-white p-1.5 rounded-full shadow border border-gray-200 text-gray-600 hover:text-blue-600">
               <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
             </button>
          </div>
        </div>

        <div class="space-y-2">
          <label class="block text-sm font-bold text-gray-700">Full Name</label>
          <input 
            id="profile-name-input"
            v-model="name"
            @input="updateName"
            type="text" 
            class="w-full border border-gray-300 rounded-lg px-4 py-2.5 focus:ring-2 focus:ring-blue-100 focus:border-blue-500 focus:outline-none transition-all"
          />
        </div>

        <div class="space-y-2">
          <label class="block text-sm font-bold text-gray-700">Bio / Headline</label>
          <textarea 
            id="profile-bio-input"
            v-model="bio"
            @input="updateBio"
            rows="3"
            class="w-full border border-gray-300 rounded-lg px-4 py-2.5 focus:ring-2 focus:ring-blue-100 focus:border-blue-500 focus:outline-none transition-all resize-none"
          ></textarea>
        </div>
      </div>

      <!-- Footer -->
      <div class="px-8 py-5 bg-gray-50 border-t border-gray-200 flex justify-end gap-3">
        <button @click="cancel" class="px-6 py-2.5 text-gray-600 font-medium hover:bg-gray-200 rounded-full transition-colors">
          Cancel
        </button>
        <button 
          id="profile-save-button" 
          @click="saveProfile" 
          :disabled="!isValid"
          :class="[
            'px-8 py-2.5 text-white font-bold rounded-full transition-all shadow-sm',
            isValid ? 'bg-blue-600 hover:bg-blue-700' : 'bg-blue-300 cursor-not-allowed'
          ]"
        >
          Save Changes
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'PROFILE_EDIT',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const name = ref('')
    const bio = ref('')

    onMounted(() => {
      // Initialize with current values
      name.value = store.profile_name
      bio.value = store.profile_bio
    })

    const isValid = computed(() => name.value.trim().length > 0)

    function updateName() {
      // Direct store update usually handled on save in traditional apps, 
      // but FSM pattern might suggest immediate update or buffered.
      // FSM Action ACT_PROFILE_EDIT_NAME effects: set profile_name directly.
      store.profile_name = name.value
    }

    function updateBio() {
      store.profile_bio = bio.value
    }

    function cancel() {
      // Revert changes if needed (not strictly required by FSM unless separate state)
      // Since we updated store directly per FSM action spec, cancelling might leave edits?
      // FSM doesn't specify revert. Assuming user wants to keep or manual revert.
      // But typically "Cancel" means discard. 
      // However, the action is ACT_PROFILE_EDIT_BACK -> to PROFILE.
      store.setCurrentPageId('PROFILE')
      router.push({ name: 'PROFILE' })
    }

    async function saveProfile() {
      if (!isValid.value) return
      store.setCurrentPageId('EDIT_PROFILE_SUCCESS')
      await router.push({ name: 'EDIT_PROFILE_SUCCESS' })
    }

    return {
      name,
      bio,
      isValid,
      updateName,
      updateBio,
      cancel,
      saveProfile
    }
  }
}
</script>