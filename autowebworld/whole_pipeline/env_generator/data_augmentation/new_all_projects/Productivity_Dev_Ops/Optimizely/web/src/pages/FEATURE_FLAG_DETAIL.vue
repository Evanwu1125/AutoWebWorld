<template>
  <div class="min-h-screen bg-gray-50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <button id="back-to-feature-flags" @click="goBack" class="flex items-center text-sm text-gray-500 hover:text-gray-700 mb-6">
        <svg class="w-4 h-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
        </svg>
        Back to Feature Flags
      </button>

      <div v-if="flag" class="bg-white shadow-lg rounded-lg overflow-hidden">
        <div id="feature-flag-header" class="bg-white border-b border-gray-200 px-8 py-6">
          <div class="flex items-center justify-between">
            <div>
              <h1 class="text-2xl font-bold text-gray-900">{{ flag.name }}</h1>
              <p class="text-gray-500 text-sm mt-1 font-mono">{{ flag.key }}</p>
            </div>
            <span :class="[
              'inline-flex items-center px-3 py-1 rounded-full text-sm font-medium',
              flag.status === 'Active' ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
            ]">
              {{ flag.status }}
            </span>
          </div>
        </div>

        <div class="p-8 grid grid-cols-1 lg:grid-cols-2 gap-12">
          <div>
            <h3 class="text-lg font-medium text-gray-900 mb-6">Rollout Configuration</h3>
            <div class="bg-gray-50 rounded-lg p-6 border border-gray-200">
              <div class="mb-8">
                <div class="flex justify-between text-sm font-medium text-gray-700 mb-2">
                  <span>Rollout Percentage</span>
                  <span class="text-blue-600">{{ rollout }}%</span>
                </div>
                <input 
                  id="feature-flag-rollout-slider"
                  type="range" 
                  v-model="rollout"
                  @input="updateRollout"
                  min="0"
                  max="100"
                  class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                >
                <p class="mt-2 text-xs text-gray-500">Percentage of users who will see this feature.</p>
              </div>
              
              <div class="flex justify-end">
                 <button 
                   id="btn-save-rollout"
                   @click="save"
                   :disabled="!isValid"
                   class="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                 >
                   Save Changes
                 </button>
              </div>
            </div>
          </div>

          <div class="space-y-6">
             <img :src="flag.image" class="w-full h-64 object-cover rounded-lg shadow-sm" alt="Feature Preview" />
             <div class="bg-white border rounded-lg p-4 shadow-sm">
               <h4 class="text-sm font-medium text-gray-900 mb-2">Details</h4>
               <dl class="grid grid-cols-2 gap-4 text-sm">
                 <div>
                   <dt class="text-gray-500">Created</dt>
                   <dd class="font-medium text-gray-900">{{ flag.created }}</dd>
                 </div>
                 <div>
                   <dt class="text-gray-500">ID</dt>
                   <dd class="font-medium text-gray-900 truncate">{{ flag.id }}</dd>
                 </div>
               </dl>
             </div>
          </div>
        </div>
      </div>
      <div v-else class="text-center py-20">
        Loading feature flag...
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted, watch } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'FEATURE_FLAG_DETAIL',
  setup() {
    const router = useRouter()
    const route = useRoute()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const flag = computed(() => {
      return dataStore.feature_flags.find(f => f.id === route.params.id)
    })
    
    const rollout = ref(0)

    onMounted(() => {
      if (flag.value) {
        signatureStore.selected_feature_flag_id = flag.value.id
        rollout.value = flag.value.rollout
      }
    })

    function updateRollout() {
      signatureStore.rollout_slider_set = true
    }

    const isValid = computed(() => {
      return signatureStore.rollout_slider_set
    })

    function save() {
      if (isValid.value) {
        // Note: FSM says go to EXPERIMENT_LAUNCHED_SUCCESS (odd, but following spec)
        signatureStore.setCurrentPageId('EXPERIMENT_LAUNCHED_SUCCESS')
        router.push({ name: 'EXPERIMENT_LAUNCHED_SUCCESS' })
      }
    }

    function goBack() {
      signatureStore.setCurrentPageId('FEATURE_FLAGS_LIST')
      router.push({ name: 'FEATURE_FLAGS_LIST' })
    }

    return {
      flag,
      rollout,
      updateRollout,
      isValid,
      save,
      goBack
    }
  }
}
</script>