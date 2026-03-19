<template>
  <div class="min-h-screen bg-gray-50 flex flex-col justify-center py-12 sm:px-6 lg:px-8">
    <div class="sm:mx-auto sm:w-full sm:max-w-md">
      <div class="bg-white py-8 px-4 shadow sm:rounded-lg sm:px-10">
        <div class="mb-6">
          <h2 class="text-xl font-bold text-gray-900">Archive Experiment</h2>
          <p class="text-sm text-gray-500 mt-1">Are you sure you want to archive this experiment? This action cannot be undone immediately.</p>
        </div>
        
        <div class="mb-6">
          <label for="input-archive-reason" class="block text-sm font-medium text-gray-700 mb-2">Reason for archiving</label>
          <input 
            id="input-archive-reason" 
            type="text" 
            v-model="reason"
            @input="updateReason"
            class="shadow-sm focus:ring-red-500 focus:border-red-500 block w-full sm:text-sm border-gray-300 rounded-md p-2 border"
            placeholder="e.g. Test complete, Invalid hypothesis..."
          >
        </div>
        
        <div class="flex space-x-3">
          <button 
            id="btn-archive-cancel"
            @click="cancel"
            class="w-full flex justify-center py-2 px-4 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none"
          >
            Cancel
          </button>
          
          <button 
            id="btn-confirm-archive"
            @click="confirm"
            :disabled="!isValid"
            class="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-red-600 hover:bg-red-700 focus:outline-none disabled:opacity-50"
          >
            Confirm Archive
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
  name: 'EXPERIMENT_ARCHIVE_CONFIRM',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const reason = ref('')

    function updateReason() {
      signatureStore.archive_reason = reason.value
    }

    const isValid = computed(() => {
      // Check both ID and reason. ID should be set from previous page.
      return signatureStore.selected_experiment_id && reason.value.length > 0
    })

    function confirm() {
      if (isValid.value) {
        signatureStore.setCurrentPageId('EXPERIMENT_ARCHIVED_SUCCESS')
        router.push({ name: 'EXPERIMENT_ARCHIVED_SUCCESS' })
      }
    }

    function cancel() {
      signatureStore.setCurrentPageId('EXPERIMENT_DETAIL')
      // Need ID to go back to detail
      router.push({ name: 'EXPERIMENT_DETAIL', params: { id: signatureStore.selected_experiment_id } })
    }

    return {
      reason,
      updateReason,
      isValid,
      confirm,
      cancel
    }
  }
}
</script>