<template>
  <div class="min-h-screen bg-gray-50 py-12">
    <div class="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="bg-white rounded-lg shadow-lg overflow-hidden">
        <div class="px-8 py-6 border-b border-gray-200">
          <h1 class="text-2xl font-bold text-gray-900">Create New Experiment</h1>
          <p class="mt-1 text-sm text-gray-500">Step 1: Define Experiment Basics</p>
        </div>
        
        <div class="p-8 space-y-6">
          <!-- Name Input -->
          <div>
            <label for="input-experiment-name" class="block text-sm font-medium text-gray-700">Experiment Name</label>
            <input 
              id="input-experiment-name"
              v-model="name"
              type="text"
              @input="updateName"
              class="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm p-2 border"
              placeholder="e.g., Homepage Redesign Q1"
            >
          </div>

          <!-- URL Input -->
          <div>
            <label for="input-experiment-url" class="block text-sm font-medium text-gray-700">Editor URL</label>
            <input 
              id="input-experiment-url"
              v-model="url"
              type="text"
              @input="updateUrl"
              class="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm p-2 border"
              placeholder="https://www.example.com"
            >
          </div>

          <!-- Type Dropdown -->
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Experiment Type</label>
            <div class="relative" id="experiment-type-dropdown">
              <button 
                type="button"
                @click="toggleTypeDropdown"
                class="bg-white relative w-full border border-gray-300 rounded-md shadow-sm pl-3 pr-10 py-2 text-left cursor-default focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
              >
                <span class="block truncate">{{ typeLabel }}</span>
                <span class="absolute inset-y-0 right-0 flex items-center pr-2 pointer-events-none">
                  <svg class="h-5 w-5 text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                    <path fill-rule="evenodd" d="M10 3a1 1 0 01.707.293l3 3a1 1 0 01-1.414 1.414L10 5.414 7.707 7.707a1 1 0 01-1.414-1.414l3-3A1 1 0 0110 3zm-3.707 9.293a1 1 0 011.414 0L10 14.586l2.293-2.293a1 1 0 011.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clip-rule="evenodd" />
                  </svg>
                </span>
              </button>

              <div v-if="typeDropdownOpen" class="absolute z-10 mt-1 w-full bg-white shadow-lg max-h-60 rounded-md py-1 text-base ring-1 ring-black ring-opacity-5 overflow-auto focus:outline-none sm:text-sm">
                <div 
                  id="experiment-type-ab" 
                  @click="selectType('a_b', 'A/B Test')"
                  class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-blue-50 text-gray-900"
                >
                  <div class="flex items-center">
                    <span class="font-normal block truncate">A/B Test</span>
                  </div>
                </div>
                <div 
                  id="experiment-type-mvt" 
                  @click="selectType('multivariate', 'Multivariate Test')"
                  class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-blue-50 text-gray-900"
                >
                   <span class="font-normal block truncate">Multivariate Test</span>
                </div>
                <div 
                  id="experiment-type-personalization" 
                  @click="selectType('personalization', 'Personalization')"
                  class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-blue-50 text-gray-900"
                >
                   <span class="font-normal block truncate">Personalization</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Footer Actions -->
        <div class="bg-gray-50 px-8 py-6 flex justify-between items-center">
          <button 
            id="create-experiment-back"
            @click="goBack"
            class="text-sm text-gray-600 hover:text-gray-900 font-medium"
          >
            Cancel
          </button>
          <button 
            id="btn-create-experiment-next"
            @click="goNext"
            :disabled="!isValid"
            class="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Create & Continue
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'EXPERIMENT_CREATE_TYPE',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const name = ref('')
    const url = ref('')
    const typeDropdownOpen = ref(false)
    const typeLabel = ref('Select type...')

    function updateName() {
      signatureStore.new_experiment_name = name.value
    }

    function updateUrl() {
      signatureStore.new_experiment_url = url.value
    }

    function toggleTypeDropdown() {
      typeDropdownOpen.value = !typeDropdownOpen.value
    }

    function selectType(value, label) {
      signatureStore.new_experiment_type = value
      typeLabel.value = label
      typeDropdownOpen.value = false
    }

    const isValid = computed(() => {
      return name.value.length > 0 && url.value.length > 0 && signatureStore.new_experiment_type
    })

    function goNext() {
      if (isValid.value) {
        signatureStore.setCurrentPageId('EXPERIMENT_EDIT_VARIATIONS')
        router.push({ name: 'EXPERIMENT_EDIT_VARIATIONS' })
      }
    }

    function goBack() {
      signatureStore.setCurrentPageId('EXPERIMENTS_LIST')
      router.push({ name: 'EXPERIMENTS_LIST' })
    }

    return {
      name,
      url,
      typeDropdownOpen,
      typeLabel,
      updateName,
      updateUrl,
      toggleTypeDropdown,
      selectType,
      isValid,
      goNext,
      goBack
    }
  }
}
</script>