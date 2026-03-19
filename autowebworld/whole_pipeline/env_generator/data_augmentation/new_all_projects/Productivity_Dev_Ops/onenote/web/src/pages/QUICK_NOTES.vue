<template>
  <div class="min-h-screen bg-yellow-50 flex flex-col">
    <!-- Navbar -->
    <header class="bg-yellow-400 shadow-md z-20 sticky top-0">
      <div class="max-w-4xl mx-auto px-4 py-4 flex justify-between items-center text-yellow-900">
        <div class="flex items-center gap-4">
          <button 
            id="back-home-from-quick-notes" 
            @click="goHome" 
            class="hover:bg-yellow-500 hover:text-white p-2 rounded-full transition"
          >
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
          </button>
          <h1 class="text-2xl font-bold flex items-center gap-2">
            <span>📝</span> Quick Notes
          </h1>
        </div>
      </div>
    </header>

    <!-- Main Content -->
    <main class="flex-1 max-w-4xl mx-auto w-full px-4 py-8">
      
      <!-- Quick Notes Grid -->
      <div 
        id="quick-notes-list-container"
        class="grid grid-cols-2 md:grid-cols-3 gap-4"
      >
      <div id="quick-notes-list" class="space-y-4">
        <div
          v-for="qn in dataStore.quick_notes"
          :key="qn.id"
          class="group bg-yellow-100 hover:bg-yellow-200 transition-colors p-4 rounded-lg shadow-sm cursor-pointer border border-yellow-200 relative overflow-hidden quick-note-row-visible"
          :class="[`data-note_id-${qn.id}`]"
          :data-id="qn.id"
          @click="openQuickNote(qn)"
        >
          <!-- Paper Texture Effect -->
          <div class="absolute inset-0 opacity-10 pointer-events-none" style="background-image: radial-gradient(#a16207 1px, transparent 1px); background-size: 20px 20px;"></div>
          
          <h3 class="font-bold text-yellow-900 mb-2 truncate relative z-10">{{ qn.title }}</h3>
          <p class="text-sm text-yellow-800 line-clamp-4 relative z-10 leading-relaxed font-handwriting">{{ qn.body }}</p>
          
          <div class="mt-4 pt-2 border-t border-yellow-300/50 flex justify-between items-center text-xs text-yellow-700 relative z-10">
            <span>{{ qn.created_at }}</span>
          </div>
        </div>
      
      </div>
      </div>

    </main>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'QUICK_NOTES',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const openQuickNote = async (note) => {
      store.selected_quick_note_id = note.id

      // Load into editor as a regular note
      store.selected_page_id = note.id // Treat as page ID for editor logic
      store.note_title = note.title
      store.note_body = note.body
      store.note_tag_color = 'yellow'

      store.current_page_id = 'NOTE_EDITOR'
      await router.push({ name: 'NOTE_EDITOR' })
    }

    const goHome = async () => {
      store.current_page_id = 'HOME'
      await router.push({ name: 'HOME' })
    }

    return {
      store,
      dataStore,
      openQuickNote,
      goHome
    }
  }
}
</script>

<style scoped>
.font-handwriting {
  font-family: 'Courier New', Courier, monospace; /* Fallback for handwritten feel */
}
</style>