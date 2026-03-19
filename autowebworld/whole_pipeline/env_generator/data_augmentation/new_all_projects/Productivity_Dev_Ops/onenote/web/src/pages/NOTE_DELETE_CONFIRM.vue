<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center p-4">
    <div class="bg-white rounded-2xl shadow-xl w-full max-w-sm p-8 text-center">
      <div class="mb-6 flex justify-center">
        <div class="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center text-red-500">
          <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path></svg>
        </div>
      </div>
      
      <h2 class="text-xl font-bold text-gray-900 mb-2">Delete this note?</h2>
      <p class="text-gray-500 mb-6">
        This action cannot be undone. Are you sure you want to permanently delete this page?
      </p>

      <!-- Confirmation Checkbox -->
      <div class="flex items-center justify-center gap-3 mb-8">
        <div 
          id="note-delete-confirm-checkbox"
          @click="toggleConfirm"
          class="w-5 h-5 border-2 rounded cursor-pointer flex items-center justify-center transition-colors"
          :class="isChecked ? 'bg-red-500 border-red-500' : 'border-gray-300'"
        >
          <svg v-if="isChecked" class="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
        </div>
        <span class="text-sm text-gray-700">Yes, I understand</span>
      </div>

      <div class="flex flex-col gap-3">
        <button 
          id="confirm-delete-note-button"
          @click="confirmDelete"
          :disabled="!isChecked"
          class="w-full bg-red-600 hover:bg-red-700 disabled:bg-gray-300 disabled:cursor-not-allowed text-white font-bold py-3 rounded-lg shadow-md transition-all"
        >
          Delete Permanently
        </button>
        
        <button 
          id="cancel-delete-note-button"
          @click="cancelDelete"
          class="w-full bg-white hover:bg-gray-50 text-gray-700 font-bold py-3 rounded-lg border border-gray-200 transition-colors"
        >
          Cancel
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'NOTE_DELETE_CONFIRM',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    
    const isChecked = ref(false)

    const toggleConfirm = () => {
      isChecked.value = !isChecked.value
      if (isChecked.value) {
        store.delete_confirmation_checked = true
      }
    }

    const confirmDelete = async () => {
      if (isChecked.value) {
        store.setCurrentPageId('NOTE_DELETE_SUCCESS')
        await router.push({ name: 'NOTE_DELETE_SUCCESS' })
      }
    }

    const cancelDelete = async () => {
      store.delete_confirmation_checked = null
      store.setCurrentPageId('NOTE_EDITOR')
      await router.push({ name: 'NOTE_EDITOR' })
    }

    return {
      isChecked,
      toggleConfirm,
      confirmDelete,
      cancelDelete
    }
  }
}
</script>