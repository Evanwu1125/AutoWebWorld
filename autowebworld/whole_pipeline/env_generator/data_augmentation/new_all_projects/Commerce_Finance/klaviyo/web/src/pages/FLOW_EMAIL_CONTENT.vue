<template>
  <div class="min-h-screen bg-slate-50 py-12">
    <div class="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="bg-white shadow rounded-lg overflow-hidden">
        <div class="px-4 py-5 sm:p-6 space-y-6">
          <h2 class="text-xl font-bold text-slate-900">Configure Email</h2>
          
          <div>
            <label for="flow-email-subject-input" class="block text-sm font-medium text-slate-700">Subject Line</label>
            <input 
              type="text" 
              id="flow-email-subject-input"
              v-model="inputSubject"
              @input="handleSubjectInput"
              class="mt-1 shadow-sm focus:ring-blue-500 focus:border-blue-500 block w-full sm:text-sm border-slate-300 rounded-md py-2 px-3"
              placeholder="e.g. Welcome to the family!"
            />
          </div>

          <div>
             <label for="flow-email-body-editor" class="block text-sm font-medium text-slate-700">Email Content</label>
             <textarea 
               id="flow-email-body-editor"
               v-model="inputBody"
               @input="handleBodyInput"
               rows="6"
               class="mt-1 shadow-sm focus:ring-blue-500 focus:border-blue-500 block w-full sm:text-sm border-slate-300 rounded-md p-3"
               placeholder="Hi there..."
             ></textarea>
          </div>
        </div>
        <div class="px-4 py-4 bg-slate-50 border-t border-slate-200 sm:px-6 flex justify-between">
          <button 
            id="back-flow-trigger"
            @click="goBack"
            class="inline-flex justify-center py-2 px-4 border border-slate-300 shadow-sm text-sm font-medium rounded-md text-slate-700 bg-white hover:bg-slate-50 focus:outline-none"
          >
            Back
          </button>
          <button 
            id="btn-flow-email-continue"
            @click="goContinue"
            :disabled="!isValid"
            class="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none disabled:opacity-50"
          >
            Next: Review
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
  name: 'FLOW_EMAIL_CONTENT',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const inputSubject = ref('')
    const inputBody = ref('')

    function handleSubjectInput() {
      store.flow_email_subject = `Flow ${inputSubject.value}`
    }

    function handleBodyInput() {
      if (inputBody.value.length > 0) {
        store.flow_email_body_has_text = true
      } else {
        store.flow_email_body_has_text = false
      }
    }

    const isValid = computed(() => {
      return store.flow_email_subject && store.flow_email_subject.length > 0 &&
             store.flow_email_body_has_text === true
    })

    async function goBack() {
      store.setCurrentPageId('FLOW_TRIGGER_SETUP')
      await router.push({ name: 'FLOW_TRIGGER_SETUP' })
    }

    async function goContinue() {
      if (!isValid.value) return
      store.setCurrentPageId('FLOW_REVIEW_ACTIVATE')
      await router.push({ name: 'FLOW_REVIEW_ACTIVATE' })
    }

    return {
      inputSubject,
      inputBody,
      handleSubjectInput,
      handleBodyInput,
      isValid,
      goBack,
      goContinue
    }
  }
}
</script>