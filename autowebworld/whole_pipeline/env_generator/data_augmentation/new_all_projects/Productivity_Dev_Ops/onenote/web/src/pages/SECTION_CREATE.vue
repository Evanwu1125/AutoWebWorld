<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center p-4">
    <div class="bg-white rounded-2xl shadow-xl w-full max-w-lg p-8">
      <div class="flex items-center justify-between mb-8">
        <h2 class="text-3xl font-bold text-gray-900">Add New Section</h2>
        <button 
          id="back-section-list" 
          @click="goBack" 
          class="text-gray-400 hover:text-gray-600 transition"
        >
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
        </button>
      </div>

      <div class="space-y-6">
        <!-- Name Input -->
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-1">Section Name</label>
          <input 
            id="new-section-name-input"
            type="text"
            v-model="sectionName"
            @input="updateName"
            placeholder="e.g. Brainstorming"
            class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500 outline-none transition"
          />
        </div>

        <!-- Info Text -->
        <p class="text-sm text-gray-500">
          This section will be added to your current notebook.
        </p>

        <!-- Submit Button -->
        <button 
          id="create-section-submit"
          @click="submitCreate"
          :disabled="!isValid"
          class="w-full bg-purple-600 hover:bg-purple-700 disabled:bg-gray-300 disabled:cursor-not-allowed text-white font-bold py-3 rounded-lg shadow-md transition-all transform active:scale-95"
        >
          Create Section
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
  name: 'SECTION_CREATE',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    
    const sectionName = ref('')

    const isValid = computed(() => {
      return sectionName.value.length > 0
    })

    const updateName = () => {
      store.new_section_name = sectionName.value
    }

    const submitCreate = async () => {
      if (isValid.value) {
        store.setCurrentPageId('SECTION_CREATE_SUCCESS')
        await router.push({ name: 'SECTION_CREATE_SUCCESS' })
      }
    }

    const goBack = async () => {
      store.setCurrentPageId('SECTION_LIST')
      await router.push({ name: 'SECTION_LIST' })
    }

    return {
      sectionName,
      isValid,
      updateName,
      submitCreate,
      goBack
    }
  }
}
</script>