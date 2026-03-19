<template>
  <div class="min-h-screen bg-slate-50 py-12">
    <div class="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="bg-white shadow rounded-lg overflow-hidden">
        <div class="px-4 py-5 sm:p-6 space-y-6">
          <h2 class="text-xl font-bold text-slate-900">SMS Message</h2>
          
          <div class="relative">
            <label for="sms-body-input" class="block text-sm font-medium text-slate-700 mb-2">Message Body</label>
            <textarea 
              id="sms-body-input"
              v-model="inputBody"
              @input="handleInput"
              rows="4"
              class="shadow-sm focus:ring-purple-500 focus:border-purple-500 block w-full sm:text-sm border-slate-300 rounded-md p-3"
              placeholder="Type your SMS message..."
              maxlength="160"
            ></textarea>
            <div class="mt-1 text-right text-xs text-slate-500">
              {{ inputBody.length }}/160 characters
            </div>
          </div>
          
          <!-- SMS Preview (Decorative) -->
          <div class="bg-slate-100 p-4 rounded-xl max-w-sm mx-auto mt-8 border border-slate-200">
            <div class="bg-white rounded-lg p-3 shadow-sm relative">
              <div class="text-sm text-slate-800 break-words">{{ inputBody || 'Message preview...' }}</div>
              <div class="text-[10px] text-slate-400 mt-1 text-right">Just now</div>
              
              <!-- Tail -->
              <div class="absolute bottom-[-6px] left-[-6px] w-4 h-4 bg-white transform rotate-45 border-b border-l border-slate-200"></div>
            </div>
          </div>

        </div>
        <div class="px-4 py-4 bg-slate-50 border-t border-slate-200 sm:px-6 flex justify-between">
          <button 
            id="back-sms-recipients"
            @click="goBack"
            class="inline-flex justify-center py-2 px-4 border border-slate-300 shadow-sm text-sm font-medium rounded-md text-slate-700 bg-white hover:bg-slate-50 focus:outline-none"
          >
            Back
          </button>
          <button 
            id="btn-sms-content-continue"
            @click="goContinue"
            :disabled="!isValid"
            class="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-purple-600 hover:bg-purple-700 focus:outline-none disabled:opacity-50"
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

export default {
  name: 'CREATE_SMS_CAMPAIGN_CONTENT',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const inputBody = ref('')

    function handleInput() {
      if (inputBody.value.length > 0) {
        store.sms_body_has_text = true
      } else {
        store.sms_body_has_text = false
      }
    }

    const isValid = computed(() => store.sms_body_has_text === true)

    async function goBack() {
      store.setCurrentPageId('CREATE_SMS_CAMPAIGN_RECIPIENTS')
      await router.push({ name: 'CREATE_SMS_CAMPAIGN_RECIPIENTS' })
    }

    async function goContinue() {
      if (!isValid.value) return
      store.setCurrentPageId('CREATE_SMS_CAMPAIGN_REVIEW_SCHEDULE')
      await router.push({ name: 'CREATE_SMS_CAMPAIGN_REVIEW_SCHEDULE' })
    }

    return {
      inputBody,
      handleInput,
      isValid,
      goBack,
      goContinue
    }
  }
}
</script>