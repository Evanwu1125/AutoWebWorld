<template>
  <div class="min-h-screen bg-slate-50 py-12">
    <div class="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="bg-white shadow rounded-lg overflow-hidden">
        <div class="px-4 py-5 sm:p-6 space-y-6">
          <div class="flex items-center space-x-2 mb-6">
            <span class="px-2 py-1 bg-purple-100 text-purple-800 text-xs font-bold uppercase rounded">SMS</span>
            <h2 class="text-xl font-bold text-slate-900">Campaign Basics</h2>
          </div>
          
          <div>
            <label for="input-sms-campaign-name" class="block text-sm font-medium text-slate-700">Campaign Name</label>
            <input 
              type="text" 
              id="input-sms-campaign-name"
              v-model="inputName"
              @input="handleNameInput"
              class="mt-1 shadow-sm focus:ring-purple-500 focus:border-purple-500 block w-full sm:text-sm border-slate-300 rounded-md py-2 px-3"
            />
          </div>

          <div>
            <label for="input-sms-sender-id" class="block text-sm font-medium text-slate-700">Sender ID (e.g. BrandName)</label>
            <input 
              type="text" 
              id="input-sms-sender-id"
              v-model="inputSender"
              @input="handleSenderInput"
              class="mt-1 shadow-sm focus:ring-purple-500 focus:border-purple-500 block w-full sm:text-sm border-slate-300 rounded-md py-2 px-3"
            />
          </div>
        </div>
        <div class="px-4 py-4 bg-slate-50 border-t border-slate-200 sm:px-6 flex justify-between">
          <button 
            id="back-sms-channel"
            @click="goBack"
            class="inline-flex justify-center py-2 px-4 border border-slate-300 shadow-sm text-sm font-medium rounded-md text-slate-700 bg-white hover:bg-slate-50 focus:outline-none"
          >
            Back
          </button>
          <button 
            id="btn-sms-basics-continue"
            @click="goContinue"
            :disabled="!isValid"
            class="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-purple-600 hover:bg-purple-700 focus:outline-none disabled:opacity-50"
          >
            Continue
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
  name: 'CREATE_SMS_CAMPAIGN_BASICS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const inputName = ref('')
    const inputSender = ref('')

    function handleNameInput() {
      store.sms_campaign_name = `SMS ${inputName.value}`
    }

    function handleSenderInput() {
      store.sms_sender_id = inputSender.value
    }

    const isValid = computed(() => {
      return store.sms_campaign_name && store.sms_campaign_name.length > 0 &&
             store.sms_sender_id && store.sms_sender_id.length > 0
    })

    async function goBack() {
      store.setCurrentPageId('CREATE_CAMPAIGN_CHANNEL')
      await router.push({ name: 'CREATE_CAMPAIGN_CHANNEL' })
    }

    async function goContinue() {
      if (!isValid.value) return
      store.setCurrentPageId('CREATE_SMS_CAMPAIGN_RECIPIENTS')
      await router.push({ name: 'CREATE_SMS_CAMPAIGN_RECIPIENTS' })
    }

    return {
      inputName,
      inputSender,
      handleNameInput,
      handleSenderInput,
      isValid,
      goBack,
      goContinue
    }
  }
}
</script>