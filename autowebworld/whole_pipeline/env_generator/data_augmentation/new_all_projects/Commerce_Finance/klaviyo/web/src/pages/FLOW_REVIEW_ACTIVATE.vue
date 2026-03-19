<template>
  <div class="min-h-screen bg-slate-50 py-12">
    <div class="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="bg-white shadow rounded-lg overflow-hidden">
        <div class="px-4 py-5 sm:p-6">
          <h2 class="text-2xl font-bold text-slate-900 mb-6">Review & Activate</h2>
          
          <div class="bg-slate-50 p-6 rounded-lg border border-slate-200 mb-8">
            <h4 class="font-bold text-slate-800 mb-4">Flow Summary</h4>
            <div class="space-y-2 text-sm text-slate-600">
              <div class="flex justify-between">
                <span>Trigger:</span>
                <span class="font-medium text-slate-900 capitalize">{{ store.flow_trigger_type }}</span>
              </div>
              <div class="flex justify-between">
                <span>Subject:</span>
                <span class="font-medium text-slate-900">{{ store.flow_email_subject }}</span>
              </div>
              <div class="flex justify-between">
                <span>Content:</span>
                <span class="font-medium text-emerald-600">Configured</span>
              </div>
            </div>
          </div>

          <div class="text-center">
             <p class="text-slate-500 mb-4">Ready to turn this on?</p>
             <button 
                id="btn-activate-flow"
                @click="activateFlow"
                class="w-full sm:w-auto inline-flex justify-center py-3 px-8 border border-transparent shadow-sm text-base font-bold rounded-md text-white bg-emerald-600 hover:bg-emerald-700 focus:outline-none"
             >
                Activate Flow
             </button>
          </div>

        </div>
        <div class="px-4 py-4 bg-slate-50 border-t border-slate-200 sm:px-6">
          <button 
            id="back-flow-email"
            @click="goBack"
            class="text-sm font-medium text-slate-500 hover:text-slate-900"
          >
            Back to edit content
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'FLOW_REVIEW_ACTIVATE',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    async function goBack() {
      store.setCurrentPageId('FLOW_EMAIL_CONTENT')
      await router.push({ name: 'FLOW_EMAIL_CONTENT' })
    }

    async function activateFlow() {
      store.flow_is_activated = true
      store.setCurrentPageId('FLOW_CREATED_SUCCESS')
      await router.push({ name: 'FLOW_CREATED_SUCCESS' })
    }

    return {
      store,
      goBack,
      activateFlow
    }
  }
}
</script>