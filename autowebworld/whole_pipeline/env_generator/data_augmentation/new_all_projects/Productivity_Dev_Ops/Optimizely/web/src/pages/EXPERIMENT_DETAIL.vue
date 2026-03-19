<template>
  <div class="min-h-screen bg-gray-50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <!-- Breadcrumb / Back -->
      <button id="back-to-experiments" @click="goBack" class="flex items-center text-sm text-gray-500 hover:text-gray-700 mb-6">
        <svg class="w-4 h-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
        </svg>
        Back to Experiments
      </button>

      <div v-if="experiment" class="bg-white shadow-lg rounded-lg overflow-hidden">
        <!-- Header -->
        <div id="experiment-header" class="bg-white border-b border-gray-200 px-8 py-6 flex justify-between items-start">
          <div>
            <div class="flex items-center gap-3 mb-2">
              <h1 class="text-2xl font-bold text-gray-900">{{ experiment.name }}</h1>
              <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                {{ experiment.status }}
              </span>
            </div>
            <p class="text-gray-500 text-sm">ID: {{ experiment.id }} • Type: {{ experiment.type }}</p>
          </div>
          <div class="flex space-x-3">
            <button id="btn-archive" @click="goToArchive" class="inline-flex items-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50">
              Archive
            </button>
            <button id="btn-edit-experiment" @click="goToEdit" class="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700">
              Edit
            </button>
          </div>
        </div>

        <!-- Tabs -->
        <div class="bg-gray-50 px-8 border-b border-gray-200">
          <nav class="-mb-px flex space-x-8">
            <button class="border-blue-500 text-blue-600 whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm">
              Overview
            </button>
            <button id="tab-targeting" @click="goToTargeting" class="border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm">
              Targeting
            </button>
            <button id="tab-scheduling" @click="goToScheduling" class="border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm">
              Scheduling
            </button>
          </nav>
        </div>

        <!-- Content -->
        <div class="p-8">
          <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <!-- Left: Stats -->
            <div class="lg:col-span-2 space-y-6">
              <div class="bg-white border rounded-lg p-6">
                <h3 class="text-lg font-medium text-gray-900 mb-4">Performance</h3>
                <div class="grid grid-cols-2 gap-4">
                  <div class="bg-blue-50 p-4 rounded-lg">
                    <div class="text-sm text-blue-600 font-medium">Total Visitors</div>
                    <div class="mt-1 text-2xl font-semibold text-gray-900">{{ experiment.visitors.toLocaleString() }}</div>
                  </div>
                  <div class="bg-green-50 p-4 rounded-lg">
                    <div class="text-sm text-green-600 font-medium">Conversions</div>
                    <div class="mt-1 text-2xl font-semibold text-gray-900">{{ experiment.conversions }}</div>
                  </div>
                </div>
                <div class="mt-6 h-64 bg-gray-100 rounded flex items-center justify-center text-gray-400">
                  [Chart Placeholder]
                </div>
              </div>
            </div>

            <!-- Right: Details -->
            <div class="space-y-6">
              <div class="bg-white border rounded-lg p-6">
                 <img :src="experiment.image" class="w-full h-48 object-cover rounded-md mb-4" alt="Preview" />
                 <div class="space-y-3">
                   <div>
                     <span class="text-xs font-uppercase text-gray-500 tracking-wider">Created</span>
                     <p class="text-sm font-medium">{{ experiment.created }}</p>
                   </div>
                   <div>
                     <span class="text-xs font-uppercase text-gray-500 tracking-wider">Last Modified</span>
                     <p class="text-sm font-medium">{{ experiment.last_modified }}</p>
                   </div>
                 </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <div v-else class="text-center py-20">
        Loading experiment details...
      </div>
    </div>
  </div>
</template>

<script>
import { computed, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'EXPERIMENT_DETAIL',
  setup() {
    const router = useRouter()
    const route = useRoute()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const experiment = computed(() => {
      const id = route.params.id
      return dataStore.experiments.find(e => e.id === id)
    })

    onMounted(() => {
      if (experiment.value) {
        signatureStore.selected_experiment_id = experiment.value.id
      }
    })

    function goBack() {
      signatureStore.setCurrentPageId('EXPERIMENTS_LIST')
      router.push({ name: 'EXPERIMENTS_LIST' })
    }

    function goToEdit() {
      signatureStore.setCurrentPageId('EXPERIMENT_EDIT_VARIATIONS')
      router.push({ name: 'EXPERIMENT_EDIT_VARIATIONS' })
    }

    function goToTargeting() {
      signatureStore.setCurrentPageId('EXPERIMENT_EDIT_TARGETING')
      router.push({ name: 'EXPERIMENT_EDIT_TARGETING' })
    }

    function goToScheduling() {
      signatureStore.setCurrentPageId('EXPERIMENT_SCHEDULE')
      router.push({ name: 'EXPERIMENT_SCHEDULE' })
    }
    
    function goToArchive() {
      signatureStore.setCurrentPageId('EXPERIMENT_ARCHIVE_CONFIRM')
      router.push({ name: 'EXPERIMENT_ARCHIVE_CONFIRM' })
    }

    return {
      experiment,
      goBack,
      goToEdit,
      goToTargeting,
      goToScheduling,
      goToArchive
    }
  }
}
</script>