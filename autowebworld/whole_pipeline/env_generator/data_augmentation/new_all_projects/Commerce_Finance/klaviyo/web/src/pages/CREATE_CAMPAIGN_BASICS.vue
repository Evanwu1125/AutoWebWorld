<template>
  <div class="min-h-screen bg-slate-50 py-12">
    <div class="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
      <!-- Steps Indicator -->
      <nav aria-label="Progress" class="mb-12">
        <ol role="list" class="space-y-4 md:flex md:space-y-0 md:space-x-8">
          <li class="md:flex-1">
            <div class="group pl-4 py-2 border-l-4 border-blue-600 flex flex-col border-t-0 md:pl-0 md:pt-4 md:pb-0 md:border-l-0 md:border-t-4">
              <span class="text-xs text-blue-600 font-semibold tracking-wide uppercase">Step 1</span>
              <span class="text-sm font-medium">Campaign Info</span>
            </div>
          </li>
          <li class="md:flex-1">
            <div class="group pl-4 py-2 border-l-4 border-slate-200 flex flex-col border-t-0 md:pl-0 md:pt-4 md:pb-0 md:border-l-0 md:border-t-4">
              <span class="text-xs text-slate-500 font-semibold tracking-wide uppercase">Step 2</span>
              <span class="text-sm font-medium">Recipients</span>
            </div>
          </li>
          <li class="md:flex-1">
            <div class="group pl-4 py-2 border-l-4 border-slate-200 flex flex-col border-t-0 md:pl-0 md:pt-4 md:pb-0 md:border-l-0 md:border-t-4">
              <span class="text-xs text-slate-500 font-semibold tracking-wide uppercase">Step 3</span>
              <span class="text-sm font-medium">Content</span>
            </div>
          </li>
        </ol>
      </nav>

      <div class="bg-white shadow rounded-lg overflow-hidden">
        <div class="px-4 py-5 sm:p-6 space-y-6">
          <h2 class="text-xl font-bold text-slate-900 mb-6">Campaign Basics</h2>
          
          <!-- Campaign Name -->
          <div>
            <label for="input-campaign-name" class="block text-sm font-medium text-slate-700">Campaign Name</label>
            <div class="mt-1">
              <input 
                type="text" 
                id="input-campaign-name"
                v-model="inputName"
                @input="handleNameInput"
                class="shadow-sm focus:ring-blue-500 focus:border-blue-500 block w-full sm:text-sm border-slate-300 rounded-md py-2 px-3"
                placeholder="e.g. Monthly Newsletter"
              />
            </div>
          </div>

          <!-- Subject Line -->
          <div>
            <label for="input-subject-line" class="block text-sm font-medium text-slate-700">Subject Line</label>
            <div class="mt-1">
              <input 
                type="text" 
                id="input-subject-line"
                v-model="inputSubject"
                @input="handleSubjectInput"
                class="shadow-sm focus:ring-blue-500 focus:border-blue-500 block w-full sm:text-sm border-slate-300 rounded-md py-2 px-3"
                placeholder="The coolest news inside"
              />
            </div>
          </div>

          <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <!-- From Name -->
            <div>
              <label for="input-from-name" class="block text-sm font-medium text-slate-700">Sender Name</label>
              <div class="mt-1">
                <input 
                  type="text" 
                  id="input-from-name"
                  v-model="inputFromName"
                  @input="handleFromNameInput"
                  class="shadow-sm focus:ring-blue-500 focus:border-blue-500 block w-full sm:text-sm border-slate-300 rounded-md py-2 px-3"
                  placeholder="Your Brand"
                />
              </div>
            </div>

            <!-- From Email -->
            <div>
              <label for="input-from-email" class="block text-sm font-medium text-slate-700">Sender Email</label>
              <div class="mt-1">
                <input 
                  type="email" 
                  id="input-from-email"
                  v-model="inputFromEmail"
                  @input="handleFromEmailInput"
                  class="shadow-sm focus:ring-blue-500 focus:border-blue-500 block w-full sm:text-sm border-slate-300 rounded-md py-2 px-3"
                  placeholder="hello@brand.com"
                />
              </div>
            </div>
          </div>

        </div>
        <div class="px-4 py-4 bg-slate-50 border-t border-slate-200 sm:px-6 flex justify-between">
          <button 
            id="back-channel"
            @click="goBack"
            class="inline-flex justify-center py-2 px-4 border border-slate-300 shadow-sm text-sm font-medium rounded-md text-slate-700 bg-white hover:bg-slate-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            Back
          </button>
          <button 
            id="btn-basics-continue"
            @click="goContinue"
            :disabled="!isValid"
            class="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Continue to Recipients
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
  name: 'CREATE_CAMPAIGN_BASICS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    // Local state for inputs to allow real-time typing, 
    // but we must commit to store on input as per FSM 'type' action logic
    const inputName = ref('')
    const inputSubject = ref('')
    const inputFromName = ref('')
    const inputFromEmail = ref('')

    // FSM Effects mapping
    // "value": "Campaign {input_text}" for name
    // "value": "Subject {input_text}" for subject
    // "value": "{input_text}" for others

    function handleNameInput() {
      store.campaign_name = `Campaign ${inputName.value}` // Mapping FSM effect logic
    }

    function handleSubjectInput() {
      store.subject_line = `Subject ${inputSubject.value}`
    }

    function handleFromNameInput() {
      store.from_name = inputFromName.value
    }

    function handleFromEmailInput() {
      store.from_email = inputFromEmail.value
    }

    const isValid = computed(() => {
      // Preconditions: length_gt 0
      return store.campaign_name && store.campaign_name.length > 0 &&
             store.subject_line && store.subject_line.length > 0 &&
             store.from_email && store.from_email.length > 0
    })

    async function goBack() {
      store.setCurrentPageId('CREATE_CAMPAIGN_CHANNEL')
      await router.push({ name: 'CREATE_CAMPAIGN_CHANNEL' })
    }

    async function goContinue() {
      if (!isValid.value) return
      store.setCurrentPageId('CREATE_CAMPAIGN_RECIPIENTS')
      await router.push({ name: 'CREATE_CAMPAIGN_RECIPIENTS' })
    }

    return {
      inputName,
      inputSubject,
      inputFromName,
      inputFromEmail,
      handleNameInput,
      handleSubjectInput,
      handleFromNameInput,
      handleFromEmailInput,
      isValid,
      goBack,
      goContinue
    }
  }
}
</script>