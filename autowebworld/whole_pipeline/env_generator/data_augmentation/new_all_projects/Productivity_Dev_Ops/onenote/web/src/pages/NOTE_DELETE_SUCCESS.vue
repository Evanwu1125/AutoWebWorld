<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center p-4">
    <div class="bg-white rounded-2xl shadow-xl max-w-md w-full p-8 text-center animate-fade-in">
      <div class="mb-6 flex justify-center">
        <div class="w-20 h-20 bg-gray-100 rounded-full flex items-center justify-center text-gray-500">
          <svg class="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path></svg>
        </div>
      </div>
      
      <h2 class="text-2xl font-bold text-gray-900 mb-2">Deleted</h2>
      <p class="text-gray-600 mb-8">
        The note has been permanently deleted.
      </p>

      <div class="space-y-3">
        <button 
          id="note-delete-success-back-pages"
          @click="goPages"
          class="w-full bg-gray-800 hover:bg-gray-900 text-white font-bold py-3 rounded-lg shadow-md transition-colors"
        >
          Back to Page List
        </button>
        
        <button 
          id="note-delete-success-go-home"
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
  name: 'NOTE_DELETE_SUCCESS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const goPages = async () => {
      store.setCurrentPageId('PAGE_LIST')
      await router.push({ name: 'PAGE_LIST' })
    }

    const goHome = async () => {
      store.setCurrentPageId('HOME')
      await router.push({ name: 'HOME' })
    }

    return {
      goPages,
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