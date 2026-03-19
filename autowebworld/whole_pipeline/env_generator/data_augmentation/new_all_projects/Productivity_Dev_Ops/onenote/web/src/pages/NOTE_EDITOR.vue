<template>
  <div class="min-h-screen bg-white flex flex-col">
    <!-- Navbar / Toolbar -->
    <header class="bg-gray-50 border-b border-gray-200 sticky top-0 z-30">
      <div class="max-w-7xl mx-auto px-4 py-3 flex justify-between items-center">
        <!-- Left: Navigation -->
        <button 
          id="back-page-list" 
          @click="goBack" 
          class="flex items-center gap-2 text-gray-600 hover:text-purple-600 transition"
        >
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
          <span class="font-medium">Back</span>
        </button>

        <!-- Right: Actions -->
        <div class="flex items-center gap-3">
          <!-- Tag Color -->
          <div class="relative">
            <button 
              id="note-tag-color-dropdown"
              @click="showColorMenu = !showColorMenu"
              class="p-2 rounded-lg hover:bg-gray-200 transition"
              title="Tag Color"
            >
              <svg class="w-6 h-6" :class="currentColorClass" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M17.707 9.293a1 1 0 010 1.414l-7 7a1 1 0 01-1.414 0l-7-7A.997.997 0 012 10V5a3 3 0 013-3h5c.256 0 .512.098.707.293l7 7zM5 6a1 1 0 100-2 1 1 0 000 2z" clip-rule="evenodd"></path></svg>
            </button>
            <!-- Dropdown -->
            <div v-if="showColorMenu" class="absolute right-0 mt-2 w-32 bg-white rounded-lg shadow-xl border border-gray-100 py-1 z-40">
              <div id="note-tag-color-yellow" @click="selectColor('yellow')" class="px-4 py-2 hover:bg-yellow-50 cursor-pointer flex items-center gap-2">
                <span class="w-3 h-3 bg-yellow-500 rounded-full"></span> Yellow
              </div>
              <div id="note-tag-color-blue" @click="selectColor('blue')" class="px-4 py-2 hover:bg-blue-50 cursor-pointer flex items-center gap-2">
                <span class="w-3 h-3 bg-blue-500 rounded-full"></span> Blue
              </div>
              <div id="note-tag-color-pink" @click="selectColor('pink')" class="px-4 py-2 hover:bg-pink-50 cursor-pointer flex items-center gap-2">
                <span class="w-3 h-3 bg-pink-500 rounded-full"></span> Pink
              </div>
            </div>
          </div>

          <!-- Divider -->
          <div class="h-6 w-px bg-gray-300 mx-1"></div>

          <!-- Share -->
          <button 
            id="note-share-button"
            @click="openShare"
            :disabled="!isValid"
            class="text-gray-600 hover:text-purple-600 disabled:opacity-50 disabled:cursor-not-allowed p-2 transition"
            title="Share"
          >
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z"></path></svg>
          </button>

          <!-- Review (Update) -->
          <button 
            v-if="isEditing"
            id="note-review-button"
            @click="openReview"
            class="text-gray-600 hover:text-blue-600 p-2 transition"
            title="Review Changes"
          >
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
          </button>

          <!-- Delete -->
          <button 
            v-if="isEditing"
            id="note-delete-button"
            @click="openDelete"
            class="text-gray-600 hover:text-red-600 p-2 transition"
            title="Delete"
          >
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path></svg>
          </button>

          <!-- Save / Update -->
          <button
            id="note-save-button"
            @click="saveNote"
            :disabled="!isValid"
            class="bg-purple-600 hover:bg-purple-700 disabled:bg-purple-300 text-white font-bold py-2 px-6 rounded-lg shadow transition-colors"
          >
            {{ isEditing ? 'Update' : 'Save' }}
          </button>
        </div>
      </div>
    </header>

    <!-- Editor Area -->
    <main class="flex-1 max-w-4xl mx-auto w-full px-8 py-12 flex flex-col gap-6">
      
      <!-- Date Stamp -->
      <div class="text-sm text-gray-400">
        {{ new Date().toLocaleString() }}
      </div>

      <!-- Title Input -->
      <input 
        id="note-title-input"
        type="text"
        v-model="title"
        @input="updateTitle"
        placeholder="Page Title"
        class="text-4xl font-bold text-gray-900 border-none focus:ring-0 p-0 placeholder-gray-300 w-full bg-transparent"
      />

      <!-- Body Input (Textarea) -->
      <textarea 
        id="note-body-editor"
        v-model="body"
        @input="updateBody"
        placeholder="Start typing your notes here..."
        class="flex-1 resize-none text-lg text-gray-700 leading-relaxed border-none focus:ring-0 p-0 placeholder-gray-300 bg-transparent min-h-[50vh]"
      ></textarea>

    </main>
  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'NOTE_EDITOR',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    
    const title = ref('')
    const body = ref('')
    const showColorMenu = ref(false)
    const selectedColor = ref(null)

    // Init from store
    onMounted(() => {
      title.value = store.note_title || ''
      body.value = store.note_body || ''
      selectedColor.value = store.note_tag_color
    })

    const isEditing = computed(() => !!store.selected_page_id)
    const isValid = computed(() => title.value.length > 0 && body.value.length > 0)
    
    const currentColorClass = computed(() => {
      switch(selectedColor.value) {
        case 'yellow': return 'text-yellow-500'
        case 'blue': return 'text-blue-500'
        case 'pink': return 'text-pink-500'
        default: return 'text-gray-400'
      }
    })

    // Actions
    const updateTitle = () => {
      store.note_title = title.value
    }

    const updateBody = () => {
      store.note_body = body.value
    }

    const selectColor = (color) => {
      selectedColor.value = color
      store.note_tag_color = color
      showColorMenu.value = false
    }

    const saveNote = async () => {
      if (isValid.value) {
        store.current_page_id = 'NOTE_CREATE_SUCCESS'
        await router.push({ name: 'NOTE_CREATE_SUCCESS' })
      }
    }

    const openReview = async () => {
      store.current_page_id = 'NOTE_REVIEW'
      await router.push({ name: 'NOTE_REVIEW' })
    }

    const openShare = async () => {
      store.current_page_id = 'NOTE_SHARE'
      await router.push({ name: 'NOTE_SHARE' })
    }

    const openDelete = async () => {
      store.current_page_id = 'NOTE_DELETE_CONFIRM'
      await router.push({ name: 'NOTE_DELETE_CONFIRM' })
    }

    const goBack = async () => {
      store.current_page_id = 'PAGE_LIST'
      await router.push({ name: 'PAGE_LIST' })
    }

    return {
      title,
      body,
      showColorMenu,
      isEditing,
      isValid,
      currentColorClass,
      updateTitle,
      updateBody,
      selectColor,
      saveNote,
      openReview,
      openShare,
      openDelete,
      goBack
    }
  }
}
</script>