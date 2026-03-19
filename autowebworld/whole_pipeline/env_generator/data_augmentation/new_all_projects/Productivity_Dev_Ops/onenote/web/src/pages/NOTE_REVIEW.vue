<template>
  <div class="min-h-screen bg-gray-50 flex flex-col items-center justify-center p-4">
    <div class="bg-white rounded-2xl shadow-xl w-full max-w-4xl p-8 flex flex-col h-[80vh]">
      <!-- Header -->
      <div class="flex items-center justify-between mb-8 border-b pb-4">
        <h2 class="text-3xl font-bold text-gray-900">Review Changes</h2>
        <button 
          id="back-note-editor" 
          @click="goBack" 
          class="text-gray-500 hover:text-gray-700 transition"
        >
          Cancel
        </button>
      </div>

      <!-- Comparison Area / Review Editor -->
      <div class="flex-1 overflow-y-auto pr-2 space-y-6">
        <div class="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <label class="block text-xs font-bold text-yellow-800 uppercase mb-2">Title</label>
          <input 
            id="review-note-title-input"
            type="text"
            v-model="title"
            @input="updateTitle"
            class="w-full bg-transparent text-xl font-bold border-none focus:ring-0 p-0 text-gray-900"
          />
        </div>

        <div class="bg-yellow-50 border border-yellow-200 rounded-lg p-4 min-h-[300px] flex flex-col">
          <label class="block text-xs font-bold text-yellow-800 uppercase mb-2">Body</label>
          <textarea 
            id="review-note-body-editor"
            v-model="body"
            @input="updateBody"
            class="w-full flex-1 bg-transparent text-base border-none focus:ring-0 p-0 text-gray-800 resize-none leading-relaxed"
          ></textarea>
        </div>
      </div>

      <!-- Footer Actions -->
      <div class="mt-8 pt-4 border-t flex justify-end gap-4">
        <button 
          @click="goBack"
          class="px-6 py-2 rounded-lg text-gray-600 hover:bg-gray-100 font-medium transition"
        >
          Keep Editing
        </button>
        <button 
          id="review-note-save-button"
          @click="submitUpdate"
          :disabled="!isValid"
          class="bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white font-bold py-2 px-8 rounded-lg shadow transition-colors"
        >
          Update Note
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
  name: 'NOTE_REVIEW',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    
    const title = ref('')
    const body = ref('')

    onMounted(() => {
      // Load current edits from store
      title.value = store.note_title
      body.value = store.note_body
    })

    const isValid = computed(() => title.value.length > 0 && body.value.length > 0)

    const updateTitle = () => {
      store.note_title = title.value
    }

    const updateBody = () => {
      store.note_body = body.value
    }

    const submitUpdate = async () => {
      if (isValid.value) {
        store.setCurrentPageId('NOTE_UPDATE_SUCCESS')
        await router.push({ name: 'NOTE_UPDATE_SUCCESS' })
      }
    }

    const goBack = async () => {
      store.setCurrentPageId('NOTE_EDITOR')
      await router.push({ name: 'NOTE_EDITOR' })
    }

    return {
      title,
      body,
      isValid,
      updateTitle,
      updateBody,
      submitUpdate,
      goBack
    }
  }
}
</script>