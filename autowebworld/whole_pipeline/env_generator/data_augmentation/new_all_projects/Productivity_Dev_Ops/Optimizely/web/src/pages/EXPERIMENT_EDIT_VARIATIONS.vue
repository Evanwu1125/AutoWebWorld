<template>
  <div class="min-h-screen bg-gray-50 py-12">
    <div class="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="bg-white rounded-lg shadow-lg overflow-hidden">
        <div class="px-8 py-6 border-b border-gray-200">
          <h1 class="text-2xl font-bold text-gray-900">Variations</h1>
          <p class="mt-1 text-sm text-gray-500">Step 2: Define your variants</p>
        </div>

        <div class="p-8 space-y-8">
          <!-- Variation A -->
          <div class="p-4 bg-gray-50 rounded-lg border border-gray-200">
            <div class="flex items-center mb-4">
              <div class="h-8 w-8 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center font-bold mr-3">A</div>
              <h3 class="text-lg font-medium text-gray-900">Original</h3>
            </div>
            <div>
              <label for="input-variation-a-name" class="block text-sm font-medium text-gray-700">Variation Name</label>
              <input 
                id="input-variation-a-name"
                v-model="varA"
                @input="updateVarA"
                type="text"
                class="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm p-2 border"
              >
            </div>
          </div>

          <!-- Variation B -->
          <div class="p-4 bg-white rounded-lg border border-blue-200 shadow-sm">
            <div class="flex items-center mb-4">
              <div class="h-8 w-8 rounded-full bg-green-100 text-green-600 flex items-center justify-center font-bold mr-3">B</div>
              <h3 class="text-lg font-medium text-gray-900">Variant 1</h3>
            </div>
            <div>
              <label for="input-variation-b-name" class="block text-sm font-medium text-gray-700">Variation Name</label>
              <input 
                id="input-variation-b-name"
                v-model="varB"
                @input="updateVarB"
                type="text"
                class="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm p-2 border"
                placeholder="Enter variant name"
              >
            </div>
          </div>

          <!-- Traffic Allocation -->
          <div class="pt-4 border-t border-gray-200">
            <h3 class="text-lg font-medium text-gray-900 mb-4">Traffic Allocation</h3>
            <div class="mb-8">
              <div class="flex justify-between text-sm text-gray-600 mb-2">
                <span>Original: {{ 100 - allocation }}%</span>
                <span>Variant 1: {{ allocation }}%</span>
              </div>
              <input 
                id="traffic-allocation-slider"
                type="range" 
                v-model="allocation"
                @input="updateAllocation"
                min="0" 
                max="100" 
                class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
              >
            </div>
          </div>
        </div>

        <!-- Footer -->
        <div class="bg-gray-50 px-8 py-6 flex justify-between items-center">
          <button 
            id="btn-variations-back"
            @click="goBack"
            class="text-sm text-gray-600 hover:text-gray-900 font-medium"
          >
            Back
          </button>
          <button 
            id="btn-variations-next"
            @click="goNext"
            :disabled="!isValid"
            class="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Next: Targeting
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
  name: 'EXPERIMENT_EDIT_VARIATIONS',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const varA = ref('Original')
    const varB = ref('')
    const allocation = ref(50)

    // Initialize store default
    signatureStore.variation_a_name = 'Original'

    function updateVarA() {
      signatureStore.variation_a_name = varA.value
    }

    function updateVarB() {
      signatureStore.variation_b_name = varB.value
    }

    function updateAllocation() {
      signatureStore.traffic_allocation_slider_set = true
    }

    const isValid = computed(() => {
      return varA.value.length > 0 && varB.value.length > 0 && signatureStore.traffic_allocation_slider_set
    })

    function goNext() {
      if (isValid.value) {
        signatureStore.setCurrentPageId('EXPERIMENT_EDIT_TARGETING')
        router.push({ name: 'EXPERIMENT_EDIT_TARGETING' })
      }
    }

    function goBack() {
      signatureStore.setCurrentPageId('EXPERIMENT_CREATE_TYPE')
      router.push({ name: 'EXPERIMENT_CREATE_TYPE' })
    }

    return {
      varA,
      varB,
      allocation,
      updateVarA,
      updateVarB,
      updateAllocation,
      isValid,
      goNext,
      goBack
    }
  }
}
</script>