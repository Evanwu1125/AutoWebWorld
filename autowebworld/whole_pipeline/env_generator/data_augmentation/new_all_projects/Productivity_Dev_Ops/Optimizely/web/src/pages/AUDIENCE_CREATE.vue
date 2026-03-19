<template>
  <div class="min-h-screen bg-gray-50 py-12">
    <div class="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="bg-white rounded-lg shadow-lg overflow-hidden">
        <div class="px-8 py-6 border-b border-gray-200">
          <h1 class="text-2xl font-bold text-gray-900">Create Audience</h1>
          <p class="mt-1 text-sm text-gray-500">Define a new segment of users.</p>
        </div>

        <div class="p-8 space-y-8">
          <!-- Name -->
          <div>
            <label for="input-audience-name" class="block text-sm font-medium text-gray-700">Audience Name</label>
            <input 
              id="input-audience-name"
              v-model="name"
              @input="updateName"
              type="text"
              class="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm p-2 border"
              placeholder="e.g., High Value Customers"
            >
          </div>

          <!-- Condition -->
          <div>
            <label for="input-audience-condition" class="block text-sm font-medium text-gray-700">Condition Definition</label>
            <textarea 
              id="input-audience-condition"
              v-model="condition"
              @input="updateCondition"
              rows="3"
              class="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm p-2 border"
              placeholder="Describe the condition (e.g., Users who visited pricing page)"
            ></textarea>
          </div>

          <!-- Membership Slider -->
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">Membership Duration (Days)</label>
            <div class="flex items-center justify-between text-xs text-gray-500 mb-2">
              <span>1 Day</span>
              <span>{{ duration }} Days</span>
              <span>365 Days</span>
            </div>
            <input 
              id="audience-membership-slider"
              type="range" 
              v-model="duration"
              @input="updateDuration"
              min="1"
              max="365"
              class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
            >
          </div>
        </div>

        <!-- Footer -->
        <div class="bg-gray-50 px-8 py-6 flex justify-between items-center">
          <button 
            id="btn-audience-back"
            @click="goBack"
            class="text-sm text-gray-600 hover:text-gray-900 font-medium"
          >
            Cancel
          </button>
          <button 
            id="btn-save-audience"
            @click="save"
            :disabled="!isValid"
            class="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Save Audience
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
  name: 'AUDIENCE_CREATE',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const name = ref('')
    const condition = ref('')
    const duration = ref(30)

    function updateName() {
      signatureStore.audience_name = name.value
    }

    function updateCondition() {
      signatureStore.audience_condition = condition.value
    }

    function updateDuration() {
      signatureStore.audience_membership_slider_set = true
    }

    const isValid = computed(() => {
      return name.value.length > 0 && 
             condition.value.length > 0 && 
             signatureStore.audience_membership_slider_set
    })

    function save() {
      if (isValid.value) {
        signatureStore.setCurrentPageId('AUDIENCE_SAVED_SUCCESS')
        router.push({ name: 'AUDIENCE_SAVED_SUCCESS' })
      }
    }

    function goBack() {
      signatureStore.setCurrentPageId('AUDIENCES_LIST')
      router.push({ name: 'AUDIENCES_LIST' })
    }

    return {
      name,
      condition,
      duration,
      updateName,
      updateCondition,
      updateDuration,
      isValid,
      save,
      goBack
    }
  }
}
</script>