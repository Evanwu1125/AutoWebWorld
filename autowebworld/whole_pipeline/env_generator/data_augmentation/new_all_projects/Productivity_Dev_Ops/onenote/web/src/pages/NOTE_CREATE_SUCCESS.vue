<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center p-4">
    <div class="bg-white rounded-2xl shadow-xl max-w-md w-full p-8 text-center animate-fade-in">
      <div class="mb-6 flex justify-center">
        <div class="w-20 h-20 bg-purple-100 rounded-full flex items-center justify-center text-purple-600">
          <svg class="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
        </div>
      </div>
      
      <h2 class="text-2xl font-bold text-gray-900 mb-2">Note Saved!</h2>
      <p class="text-gray-600 mb-8">
        Your note has been successfully saved to your notebook.
      </p>

      <div class="space-y-3">
        <button 
          id="note-create-success-back-editor"
          @click="goEditor"
          class="w-full bg-purple-600 hover:bg-purple-700 text-white font-bold py-3 rounded-lg shadow-md transition-colors"
        >
          Keep Editing
        </button>
        
        <button 
          id="note-create-success-go-home"
          @click="goHome"
          class="w-full bg-white hover:bg-gray-50 text-gray-700 font-bold py-3 rounded-lg border border-gray-200 transition-colors"
        >
          Return Home
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'NOTE_CREATE_SUCCESS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const goEditor = async () => {
      store.setCurrentPageId('NOTE_EDITOR')
      await router.push({ name: 'NOTE_EDITOR' })
    }

    const goHome = async () => {
      store.setCurrentPageId('HOME')
      await router.push({ name: 'HOME' })
    }

    return {
      goEditor,
      goHome
    }
  }
}
</script>

<style scoped>
.animate-fade-in {
  animation: fadeIn 0.5s ease-out;
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>