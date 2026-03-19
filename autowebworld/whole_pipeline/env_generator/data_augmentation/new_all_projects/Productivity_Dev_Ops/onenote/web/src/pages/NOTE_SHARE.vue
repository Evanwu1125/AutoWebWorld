<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center p-4">
    <div class="bg-white rounded-2xl shadow-xl w-full max-w-md p-8">
      <div class="flex items-center justify-between mb-8">
        <h2 class="text-2xl font-bold text-gray-900">Share Note</h2>
        <button 
          id="back-note-editor-from-share" 
          @click="goBack" 
          class="text-gray-400 hover:text-gray-600 transition"
        >
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
        </button>
      </div>

      <div class="space-y-6">
        <!-- Email Input -->
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-1">Email Address</label>
          <input 
            id="share-email-input"
            type="email"
            v-model="email"
            @input="updateEmail"
            placeholder="colleague@example.com"
            class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500 outline-none transition"
          />
        </div>

        <!-- Permission Selection -->
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-1">Permission</label>
          <div class="relative">
            <button 
              id="share-permission-dropdown"
              @click="showPermMenu = !showPermMenu"
              class="w-full text-left px-4 py-3 border border-gray-300 rounded-lg flex items-center justify-between hover:bg-gray-50 transition"
            >
              <span class="flex items-center gap-2 text-gray-700">
                <svg v-if="permission === 'edit'" class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"></path></svg>
                <svg v-else class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"></path></svg>
                {{ permission === 'edit' ? 'Can Edit' : 'Can View' }}
              </span>
              <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
            </button>

            <!-- Dropdown Options -->
            <div v-if="showPermMenu" class="absolute top-full left-0 right-0 mt-2 bg-white rounded-lg shadow-xl border border-gray-100 z-10 overflow-hidden">
              <div 
                id="share-permission-view" 
                @click="selectPermission('view')" 
                class="px-4 py-3 hover:bg-gray-50 cursor-pointer flex items-center gap-2 text-gray-700"
              >
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"></path></svg>
                Can View
              </div>
              <div 
                id="share-permission-edit" 
                @click="selectPermission('edit')" 
                class="px-4 py-3 hover:bg-gray-50 cursor-pointer flex items-center gap-2 text-gray-700"
              >
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"></path></svg>
                Can Edit
              </div>
            </div>
          </div>
        </div>

        <!-- Submit Button -->
        <button 
          id="share-send-link-button"
          @click="submitShare"
          :disabled="!isValid"
          class="w-full bg-purple-600 hover:bg-purple-700 disabled:bg-gray-300 disabled:cursor-not-allowed text-white font-bold py-3 rounded-lg shadow-md transition-all transform active:scale-95"
        >
          Send Invite
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
  name: 'NOTE_SHARE',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    
    const email = ref('')
    const permission = ref('view') // Default
    const showPermMenu = ref(false)

    const isValid = computed(() => email.value.length > 0 && permission.value)

    const updateEmail = () => {
      store.share_email = email.value
    }

    const selectPermission = (perm) => {
      permission.value = perm
      store.share_permission_level = perm
      showPermMenu.value = false
    }

    const submitShare = async () => {
      if (isValid.value) {
        store.setCurrentPageId('NOTE_SHARE_SUCCESS')
        await router.push({ name: 'NOTE_SHARE_SUCCESS' })
      }
    }

    const goBack = async () => {
      store.setCurrentPageId('NOTE_EDITOR')
      await router.push({ name: 'NOTE_EDITOR' })
    }

    return {
      email,
      permission,
      showPermMenu,
      isValid,
      updateEmail,
      selectPermission,
      submitShare,
      goBack
    }
  }
}
</script>