<template>
  <div class="min-h-screen bg-slate-50 py-12">
    <div class="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
      <!-- Steps Indicator -->
      <nav aria-label="Progress" class="mb-12">
        <ol role="list" class="space-y-4 md:flex md:space-y-0 md:space-x-8">
          <li class="md:flex-1">
            <div class="group pl-4 py-2 border-l-4 border-blue-600 flex flex-col border-t-0 md:pl-0 md:pt-4 md:pb-0 md:border-l-0 md:border-t-4 opacity-50">
              <span class="text-xs text-blue-600 font-semibold tracking-wide uppercase">Step 1</span>
              <span class="text-sm font-medium">Campaign Info</span>
            </div>
          </li>
          <li class="md:flex-1">
            <div class="group pl-4 py-2 border-l-4 border-blue-600 flex flex-col border-t-0 md:pl-0 md:pt-4 md:pb-0 md:border-l-0 md:border-t-4 opacity-50">
              <span class="text-xs text-blue-600 font-semibold tracking-wide uppercase">Step 2</span>
              <span class="text-sm font-medium">Recipients</span>
            </div>
          </li>
          <li class="md:flex-1">
            <div class="group pl-4 py-2 border-l-4 border-blue-600 flex flex-col border-t-0 md:pl-0 md:pt-4 md:pb-0 md:border-l-0 md:border-t-4">
              <span class="text-xs text-blue-600 font-semibold tracking-wide uppercase">Step 3</span>
              <span class="text-sm font-medium">Content</span>
            </div>
          </li>
        </ol>
      </nav>

      <div class="bg-white shadow rounded-lg overflow-hidden">
        <div class="px-4 py-5 sm:p-6 space-y-8">
          
          <!-- Template Picker -->
          <div>
            <h3 class="text-lg font-bold text-slate-900 mb-4">Select Template</h3>
            <div 
              id="template-picker-dropdown" 
              class="grid grid-cols-2 md:grid-cols-4 gap-4"
            >
              <div 
                v-for="template in templates" 
                :key="template.id"
                :class="[`option-template-${template.id.replace('template_','')}`, store.selected_email_template_id === template.id ? 'ring-2 ring-blue-500' : 'hover:ring-2 hover:ring-blue-300']"
                @click="selectTemplate(template.id)"
                class="cursor-pointer bg-slate-50 rounded-lg overflow-hidden transition-all relative group"
              >
                <img :src="template.image" :alt="template.name" class="w-full h-32 object-cover opacity-80 group-hover:opacity-100" />
                <div class="p-2 text-xs font-medium text-center truncate">{{ template.name }}</div>
                
                <!-- Checkmark -->
                <div v-if="store.selected_email_template_id === template.id" class="absolute top-2 right-2 bg-blue-500 text-white rounded-full p-1">
                  <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
                </div>
              </div>
            </div>
          </div>

          <!-- Body Editor -->
          <div>
            <h3 class="text-lg font-bold text-slate-900 mb-4">Edit Content</h3>
            <div class="border border-slate-300 rounded-lg overflow-hidden focus-within:ring-2 focus-within:ring-blue-500 focus-within:border-blue-500">
               <textarea 
                 id="email-body-editor"
                 v-model="emailBody"
                 @input="handleBodyInput"
                 rows="8" 
                 class="w-full p-4 border-none focus:ring-0 resize-none text-slate-800"
                 placeholder="Type your email content here..."
               ></textarea>
            </div>
          </div>

        </div>
        <div class="px-4 py-4 bg-slate-50 border-t border-slate-200 sm:px-6 flex justify-between">
          <button 
            id="back-recipients"
            @click="goBack"
            class="inline-flex justify-center py-2 px-4 border border-slate-300 shadow-sm text-sm font-medium rounded-md text-slate-700 bg-white hover:bg-slate-50 focus:outline-none"
          >
            Back
          </button>
          <button 
            id="btn-content-continue"
            @click="goContinue"
            :disabled="!isValid"
            class="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Review & Schedule
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
import { useDataStore } from '../stores/data'

export default {
  name: 'CREATE_CAMPAIGN_CONTENT',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const templates = computed(() => dataStore.email_templates)
    const emailBody = ref('')

    function selectTemplate(id) {
      store.selected_email_template_id = id
      store.email_body_has_text = true // Selecting template sets flag per FSM logic (implicitly via effects)
      // Actually FSM says: effect set email_body_has_text = true when selecting template.
    }

    function handleBodyInput() {
      // FSM says: type action sets email_body_has_text = true.
      if (emailBody.value.length > 0) {
        store.email_body_has_text = true
      } else {
        store.email_body_has_text = false // Logic not in FSM but sensible
      }
    }

    const isValid = computed(() => {
      // Precondition: email_body_has_text eq true
      return store.email_body_has_text === true
    })

    async function goBack() {
      store.setCurrentPageId('CREATE_CAMPAIGN_RECIPIENTS')
      await router.push({ name: 'CREATE_CAMPAIGN_RECIPIENTS' })
    }

    async function goContinue() {
      if (!isValid.value) return
      store.setCurrentPageId('CREATE_CAMPAIGN_REVIEW_SCHEDULE')
      await router.push({ name: 'CREATE_CAMPAIGN_REVIEW_SCHEDULE' })
    }

    return {
      store,
      templates,
      emailBody,
      selectTemplate,
      handleBodyInput,
      isValid,
      goBack,
      goContinue
    }
  }
}
</script>