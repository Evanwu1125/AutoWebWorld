<template>
  <div class="min-h-screen bg-slate-50 py-12">
    <div class="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="bg-white shadow rounded-lg overflow-hidden flex flex-col md:flex-row min-h-[600px]">
        
        <!-- Sidebar Controls -->
        <div class="w-full md:w-1/3 bg-slate-50 border-r border-slate-200 p-6 space-y-8">
           <h2 class="text-xl font-bold text-slate-900">Design Form</h2>
           
           <div>
             <label for="form-name-input" class="block text-sm font-medium text-slate-700">Form Name</label>
             <input 
               type="text" 
               id="form-name-input"
               v-model="inputName"
               @input="handleNameInput"
               class="mt-1 shadow-sm focus:ring-orange-500 focus:border-orange-500 block w-full sm:text-sm border-slate-300 rounded-md py-2 px-3"
             />
           </div>

           <div>
              <div class="flex items-center justify-between mb-2">
                <span class="block text-sm font-medium text-slate-700">Email Field</span>
                <button 
                  id="toggle-email-field"
                  @click="toggleEmail"
                  class="relative inline-flex flex-shrink-0 h-6 w-11 border-2 border-transparent rounded-full cursor-pointer transition-colors ease-in-out duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-orange-500"
                  :class="emailEnabled ? 'bg-orange-600' : 'bg-slate-200'"
                >
                  <span 
                    class="pointer-events-none inline-block h-5 w-5 rounded-full bg-white shadow transform ring-0 transition ease-in-out duration-200"
                    :class="emailEnabled ? 'translate-x-5' : 'translate-x-0'"
                  ></span>
                </button>
              </div>
              <p class="text-xs text-slate-500">Collect subscriber email addresses.</p>
           </div>

           <div>
             <label for="form-cta-input" class="block text-sm font-medium text-slate-700">Button Text</label>
             <input 
               type="text" 
               id="form-cta-input"
               v-model="inputCTA"
               @input="handleCTAInput"
               class="mt-1 shadow-sm focus:ring-orange-500 focus:border-orange-500 block w-full sm:text-sm border-slate-300 rounded-md py-2 px-3"
               placeholder="Subscribe"
             />
           </div>
           
           <div class="pt-8 mt-auto border-t border-slate-200 flex flex-col space-y-3">
             <button 
                id="btn-publish-form"
                @click="publishForm"
                :disabled="!isValid"
                class="w-full inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-orange-600 hover:bg-orange-700 focus:outline-none disabled:opacity-50"
             >
                Publish Form
             </button>
             <button 
                id="back-forms-list"
                @click="goBack"
                class="w-full inline-flex justify-center py-2 px-4 border border-slate-300 shadow-sm text-sm font-medium rounded-md text-slate-700 bg-white hover:bg-slate-50 focus:outline-none"
             >
                Exit
             </button>
           </div>
        </div>

        <!-- Preview Area -->
        <div class="w-full md:w-2/3 bg-slate-200 p-8 flex items-center justify-center">
           <div class="bg-white p-8 rounded-lg shadow-xl max-w-sm w-full text-center relative">
             <!-- Close button decoration -->
             <div class="absolute top-2 right-2 text-slate-300 hover:text-slate-500 cursor-pointer">
               <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
             </div>
             
             <h3 class="text-2xl font-bold text-slate-900 mb-2">Join Our Newsletter</h3>
             <p class="text-slate-600 mb-6">Get 10% off your first order!</p>
             
             <div class="space-y-3">
               <input v-if="emailEnabled" type="email" placeholder="Your email address" class="w-full border border-slate-300 rounded-md py-2 px-3" disabled />
               <button class="w-full bg-slate-900 text-white font-bold py-2 rounded-md uppercase tracking-wider text-sm">
                 {{ inputCTA || 'Subscribe' }}
               </button>
             </div>
             
             <p class="text-xs text-slate-400 mt-4">No spam, unsubscribe anytime.</p>
           </div>
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
  name: 'SIGNUP_FORM_BUILDER',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    
    const inputName = ref('')
    const inputCTA = ref('')
    const emailEnabled = ref(false)

    function handleNameInput() {
      store.form_name = `Form ${inputName.value}`
    }

    function handleCTAInput() {
      store.form_call_to_action = inputCTA.value
    }

    function toggleEmail() {
      emailEnabled.value = !emailEnabled.value
      if (emailEnabled.value) {
        store.form_field_email_enabled = true
      } else {
        store.form_field_email_enabled = false
      }
    }

    const isValid = computed(() => {
      return store.form_name && store.form_name.length > 0 &&
             store.form_field_email_enabled === true &&
             store.form_call_to_action && store.form_call_to_action.length > 0
    })

    async function goBack() {
      store.setCurrentPageId('SIGNUP_FORMS_LIST')
      await router.push({ name: 'SIGNUP_FORMS_LIST' })
    }

    async function publishForm() {
      if (!isValid.value) return
      store.setCurrentPageId('SIGNUP_FORM_PUBLISHED_SUCCESS')
      await router.push({ name: 'SIGNUP_FORM_PUBLISHED_SUCCESS' })
    }

    return {
      inputName,
      inputCTA,
      emailEnabled,
      handleNameInput,
      handleCTAInput,
      toggleEmail,
      isValid,
      goBack,
      publishForm
    }
  }
}
</script>