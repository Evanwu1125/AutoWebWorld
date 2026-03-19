<template>
  <div class="min-h-screen bg-gray-50 py-12">
    <div class="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="bg-white rounded-lg shadow-lg overflow-hidden">
        <div class="px-8 py-6 border-b border-gray-200">
          <h1 class="text-2xl font-bold text-gray-900">Targeting</h1>
          <p class="mt-1 text-sm text-gray-500">Step 3: Who should see this experiment?</p>
        </div>

        <div class="p-8 space-y-8">
          <!-- URL Targeting -->
          <div>
            <label for="input-url-pattern" class="block text-sm font-medium text-gray-700 mb-2">URL Match Pattern</label>
            <div class="mt-1 flex rounded-md shadow-sm">
              <span class="inline-flex items-center px-3 rounded-l-md border border-r-0 border-gray-300 bg-gray-50 text-gray-500 sm:text-sm">
                Simple Match
              </span>
              <input 
                type="text" 
                id="input-url-pattern" 
                v-model="urlPattern"
                @input="updateUrlPattern"
                class="flex-1 min-w-0 block w-full px-3 py-2 rounded-none rounded-r-md focus:ring-blue-500 focus:border-blue-500 sm:text-sm border-gray-300 border" 
                placeholder="https://example.com/products/*"
              >
            </div>
            <p class="mt-2 text-xs text-gray-500">Use * for wildcards.</p>
          </div>

          <!-- Audience Selection -->
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">Audience</label>
            <div class="relative" id="audience-dropdown">
              <button 
                type="button"
                @click="toggleAudienceDropdown"
                class="bg-white relative w-full border border-gray-300 rounded-md shadow-sm pl-3 pr-10 py-2 text-left cursor-default focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
              >
                <span class="block truncate">{{ audienceLabel }}</span>
                <span class="absolute inset-y-0 right-0 flex items-center pr-2 pointer-events-none">
                  <svg class="h-5 w-5 text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                    <path fill-rule="evenodd" d="M10 3a1 1 0 01.707.293l3 3a1 1 0 01-1.414 1.414L10 5.414 7.707 7.707a1 1 0 01-1.414-1.414l3-3A1 1 0 0110 3zm-3.707 9.293a1 1 0 011.414 0L10 14.586l2.293-2.293a1 1 0 011.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clip-rule="evenodd" />
                  </svg>
                </span>
              </button>

              <div v-if="audienceDropdownOpen" class="absolute z-10 mt-1 w-full bg-white shadow-lg max-h-60 rounded-md py-1 text-base ring-1 ring-black ring-opacity-5 overflow-auto focus:outline-none sm:text-sm">
                <div 
                  id="audience-option-1" 
                  @click="selectAudience('audience_any', 'Everyone')"
                  class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-blue-50 text-gray-900"
                >
                  <span class="font-normal block truncate">Everyone</span>
                </div>
                <div 
                  id="audience-option-2" 
                  @click="selectAudience('audience_returning', 'Returning Visitors')"
                  class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-blue-50 text-gray-900"
                >
                  <span class="font-normal block truncate">Returning Visitors</span>
                </div>
                <div 
                  id="audience-option-3" 
                  @click="selectAudience('audience_new', 'New Visitors')"
                  class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-blue-50 text-gray-900"
                >
                  <span class="font-normal block truncate">New Visitors</span>
                </div>
              </div>
            </div>
          </div>

          <!-- Device Checkbox -->
          <div class="flex items-start">
            <div class="flex items-center h-5">
              <input 
                id="target-mobile-checkbox" 
                type="checkbox" 
                v-model="mobileTargeting"
                @change="updateDevice"
                class="focus:ring-blue-500 h-4 w-4 text-blue-600 border-gray-300 rounded"
              >
            </div>
            <div class="ml-3 text-sm">
              <label for="target-mobile-checkbox" class="font-medium text-gray-700">Target Mobile Devices</label>
              <p class="text-gray-500">Limit this experiment to users on mobile phones and tablets.</p>
            </div>
          </div>
        </div>

        <!-- Footer -->
        <div class="bg-gray-50 px-8 py-6 flex justify-between items-center">
          <button 
            id="btn-targeting-back"
            @click="goBack"
            class="text-sm text-gray-600 hover:text-gray-900 font-medium"
          >
            Back
          </button>
          <button 
            id="btn-targeting-next"
            @click="goNext"
            :disabled="!isValid"
            class="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Next: Schedule
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
  name: 'EXPERIMENT_EDIT_TARGETING',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const urlPattern = ref('')
    const audienceDropdownOpen = ref(false)
    const audienceLabel = ref('Select audience...')
    const mobileTargeting = ref(false)

    function updateUrlPattern() {
      signatureStore.url_match_pattern = urlPattern.value
    }

    function toggleAudienceDropdown() {
      audienceDropdownOpen.value = !audienceDropdownOpen.value
    }

    function selectAudience(value, label) {
      signatureStore.selected_audience_id = value
      audienceLabel.value = label
      audienceDropdownOpen.value = false
    }

    function updateDevice() {
      signatureStore.targeting_device_checkbox_set = mobileTargeting.value
    }

    const isValid = computed(() => {
      return urlPattern.value.length > 0 && 
             signatureStore.selected_audience_id && 
             mobileTargeting.value === true
    })

    function goNext() {
      if (isValid.value) {
        signatureStore.setCurrentPageId('EXPERIMENT_SCHEDULE')
        router.push({ name: 'EXPERIMENT_SCHEDULE' })
      }
    }

    function goBack() {
      signatureStore.setCurrentPageId('EXPERIMENT_EDIT_VARIATIONS')
      router.push({ name: 'EXPERIMENT_EDIT_VARIATIONS' })
    }

    return {
      urlPattern,
      audienceDropdownOpen,
      audienceLabel,
      mobileTargeting,
      updateUrlPattern,
      toggleAudienceDropdown,
      selectAudience,
      updateDevice,
      isValid,
      goNext,
      goBack
    }
  }
}
</script>