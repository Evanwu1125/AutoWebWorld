<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <div class="bg-white shadow-sm">
       <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center">
          <h1 class="text-xl font-semibold text-gray-800">Results Overview</h1>
       </div>
    </div>

    <main class="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
      <div class="mb-6">
        <button id="back-to-dashboard" @click="goDashboard" class="text-sm text-blue-600 hover:text-blue-800 font-medium">
          &larr; Back to Dashboard
        </button>
      </div>

      <div id="results-table" class="bg-white shadow rounded-lg overflow-hidden">
        <div class="px-6 py-4 border-b border-gray-200">
          <h3 class="text-lg font-medium text-gray-900">Experiment Results</h3>
        </div>
        <ul class="divide-y divide-gray-200">
          <li v-for="exp in experiments" :key="exp.id" class="hover:bg-gray-50">
            <div 
              :class="[
                'px-6 py-4 cursor-pointer flex items-center',
                `data-id-${exp.id}`,
                'row-visible'
              ]"
              @click="openResult(exp)"
            >
              <div class="flex-shrink-0 h-12 w-12 mr-4">
                <img :src="exp.image" class="h-12 w-12 rounded-md object-cover" alt="" />
              </div>
              <div class="flex-1 grid grid-cols-1 md:grid-cols-4 gap-4 items-center">
                <div class="md:col-span-2">
                  <div class="text-sm font-medium text-gray-900">{{ exp.name }}</div>
                  <div class="text-xs text-gray-500">Type: {{ exp.type }}</div>
                </div>
                <div>
                  <div class="text-xs text-gray-500 uppercase tracking-wider">Conversions</div>
                  <div class="text-sm font-bold text-green-600">{{ exp.conversions }}</div>
                </div>
                <div class="text-right">
                  <div class="text-xs text-gray-500 uppercase tracking-wider">Visitors</div>
                  <div class="text-sm font-medium text-gray-900">{{ exp.visitors.toLocaleString() }}</div>
                </div>
              </div>
              <div class="ml-4">
                <svg class="h-5 w-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
                </svg>
              </div>
            </div>
          </li>
        </ul>
      </div>
    </main>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'RESULTS_OVERVIEW',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const experiments = computed(() => dataStore.experiments)

    function openResult(exp) {
      signatureStore.results_viewport_anchor_id = exp.id
      signatureStore.results_selected_item_id = exp.id
      
      signatureStore.setCurrentPageId('EXPERIMENT_DETAIL')
      router.push({ name: 'EXPERIMENT_DETAIL', params: { id: exp.id } })
    }

    function goDashboard() {
      signatureStore.setCurrentPageId('DASHBOARD')
      router.push({ name: 'DASHBOARD' })
    }

    return {
      experiments,
      openResult,
      goDashboard
    }
  }
}
</script>