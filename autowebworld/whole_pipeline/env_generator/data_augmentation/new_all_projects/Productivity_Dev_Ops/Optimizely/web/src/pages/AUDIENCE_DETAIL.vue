<template>
  <div class="min-h-screen bg-gray-50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <button id="back-to-audiences" @click="goBack" class="flex items-center text-sm text-gray-500 hover:text-gray-700 mb-6">
        <svg class="w-4 h-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
        </svg>
        Back to Audiences
      </button>

      <div v-if="audience" class="bg-white shadow-lg rounded-lg overflow-hidden">
        <div id="audience-header" class="bg-white border-b border-gray-200 px-8 py-6 flex justify-between items-start">
          <div>
            <h1 class="text-2xl font-bold text-gray-900">{{ audience.name }}</h1>
            <p class="text-gray-500 text-sm mt-1">ID: {{ audience.id }} • {{ audience.size.toLocaleString() }} Users</p>
          </div>
          <button id="btn-edit-audience" @click="goToEdit" class="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700">
            Edit Audience
          </button>
        </div>

        <div class="p-8 grid grid-cols-1 md:grid-cols-3 gap-8">
          <div class="md:col-span-2 space-y-6">
            <div class="bg-white border rounded-lg p-6">
              <h3 class="text-lg font-medium text-gray-900 mb-4">Definition</h3>
              <dl class="grid grid-cols-1 gap-x-4 gap-y-6 sm:grid-cols-2">
                <div>
                  <dt class="text-sm font-medium text-gray-500">Type</dt>
                  <dd class="mt-1 text-sm text-gray-900">{{ audience.type }}</dd>
                </div>
                <div>
                  <dt class="text-sm font-medium text-gray-500">Last Modified</dt>
                  <dd class="mt-1 text-sm text-gray-900">{{ audience.last_modified }}</dd>
                </div>
                <div class="sm:col-span-2">
                  <dt class="text-sm font-medium text-gray-500">Description</dt>
                  <dd class="mt-1 text-sm text-gray-900">{{ audience.description }}</dd>
                </div>
              </dl>
            </div>
          </div>

          <div class="space-y-6">
             <img :src="audience.image" class="w-full h-48 object-cover rounded-lg shadow-sm" alt="Audience Visualization" />
             <div class="bg-blue-50 p-4 rounded-lg border border-blue-100">
               <h4 class="text-sm font-medium text-blue-800">Usage</h4>
               <p class="mt-1 text-xs text-blue-600">Used in 3 experiments</p>
             </div>
          </div>
        </div>
      </div>
      <div v-else class="text-center py-20">
        Loading audience details...
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
  name: 'AUDIENCE_DETAIL',
  setup() {
    const router = useRouter()
    const route = useRoute()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const audience = computed(() => {
      return dataStore.audiences.find(a => a.id === route.params.id)
    })

    onMounted(() => {
      if (audience.value) {
        signatureStore.selected_audience_id = audience.value.id
      }
    })

    function goBack() {
      signatureStore.setCurrentPageId('AUDIENCES_LIST')
      router.push({ name: 'AUDIENCES_LIST' })
    }

    function goToEdit() {
      signatureStore.setCurrentPageId('AUDIENCE_CREATE')
      router.push({ name: 'AUDIENCE_CREATE' })
    }

    return {
      audience,
      goBack,
      goToEdit
    }
  }
}
</script>