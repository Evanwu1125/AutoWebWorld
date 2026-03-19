<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center p-4">
    <div class="bg-white rounded-2xl shadow-xl w-full max-w-lg p-8">
      <div class="flex items-center justify-between mb-8">
        <h2 class="text-3xl font-bold text-gray-900">Create Notebook</h2>
        <button 
          id="back-notebook-list" 
          @click="goBack" 
          class="text-gray-400 hover:text-gray-600 transition"
        >
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
        </button>
      </div>

      <div class="space-y-6">
        <!-- Name Input -->
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-1">Notebook Name</label>
          <input 
            id="new-notebook-name-input"
            type="text"
            v-model="notebookName"
            @input="updateName"
            placeholder="e.g. Vacation Plans"
            class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500 outline-none transition"
          />
        </div>

        <!-- Color Selection -->
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-1">Cover Color</label>
          <div class="relative">
            <button 
              id="notebook-color-dropdown"
              @click="showColorMenu = !showColorMenu"
              class="w-full text-left px-4 py-3 border border-gray-300 rounded-lg flex items-center justify-between hover:bg-gray-50 transition"
            >
              <span class="flex items-center gap-2">
                <span 
                  class="w-4 h-4 rounded-full" 
                  :class="{
                    'bg-blue-500': selectedColor === 'blue',
                    'bg-green-500': selectedColor === 'green',
                    'bg-red-500': selectedColor === 'red',
                    'bg-gray-300': !selectedColor
                  }"
                ></span>
                {{ selectedColor ? selectedColor.charAt(0).toUpperCase() + selectedColor.slice(1) : 'Select a color' }}
              </span>
              <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
            </button>

            <!-- Dropdown Options -->
            <div v-if="showColorMenu" class="absolute top-full left-0 right-0 mt-2 bg-white rounded-lg shadow-xl border border-gray-100 z-10 overflow-hidden">
              <div 
                id="notebook-color-blue" 
                @click="selectColor('blue')" 
                class="px-4 py-3 hover:bg-blue-50 cursor-pointer flex items-center gap-2"
              >
                <span class="w-4 h-4 rounded-full bg-blue-500"></span> Blue
              </div>
              <div 
                id="notebook-color-green" 
                @click="selectColor('green')" 
                class="px-4 py-3 hover:bg-green-50 cursor-pointer flex items-center gap-2"
              >
                <span class="w-4 h-4 rounded-full bg-green-500"></span> Green
              </div>
              <div 
                id="notebook-color-red" 
                @click="selectColor('red')" 
                class="px-4 py-3 hover:bg-red-50 cursor-pointer flex items-center gap-2"
              >
                <span class="w-4 h-4 rounded-full bg-red-500"></span> Red
              </div>
            </div>
          </div>
        </div>

        <!-- Submit Button -->
        <button 
          id="create-notebook-submit"
          @click="submitCreate"
          :disabled="!isValid"
          class="w-full bg-purple-600 hover:bg-purple-700 disabled:bg-gray-300 disabled:cursor-not-allowed text-white font-bold py-3 rounded-lg shadow-md transition-all transform active:scale-95"
        >
          Create Notebook
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
  name: 'NOTEBOOK_CREATE',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    
    const notebookName = ref('')
    const selectedColor = ref(null)
    const showColorMenu = ref(false)

    const isValid = computed(() => {
      return notebookName.value.length > 0 && selectedColor.value
    })

    const updateName = () => {
      store.new_notebook_name = notebookName.value
    }

    const selectColor = (color) => {
      selectedColor.value = color
      store.new_notebook_color = color
      showColorMenu.value = false
    }

    const submitCreate = async () => {
      if (isValid.value) {
        store.setCurrentPageId('SECTION_CREATE_SUCCESS')
        await router.push({ name: 'SECTION_CREATE_SUCCESS' })
      }
    }

    const goBack = async () => {
      store.setCurrentPageId('NOTEBOOK_LIST')
      await router.push({ name: 'NOTEBOOK_LIST' })
    }

    return {
      notebookName,
      selectedColor,
      showColorMenu,
      isValid,
      updateName,
      selectColor,
      submitCreate,
      goBack
    }
  }
}
</script>