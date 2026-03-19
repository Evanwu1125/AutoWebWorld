<template>
  <div class="min-h-screen bg-slate-50 py-12">
    <div class="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="bg-white shadow rounded-lg overflow-hidden">
        <div class="px-4 py-5 sm:p-6">
          <h2 class="text-2xl font-bold text-slate-900 mb-6">Review & Schedule SMS</h2>
          
          <div class="space-y-4 mb-8">
             <div class="bg-purple-50 rounded-lg p-4 border border-purple-100">
                <ul class="text-sm text-purple-800 space-y-1">
                  <li><strong>Name:</strong> {{ store.sms_campaign_name }}</li>
                  <li><strong>Sender:</strong> {{ store.sms_sender_id }}</li>
                </ul>
             </div>
          </div>

          <div class="space-y-4">
            <h3 class="text-lg font-medium text-slate-900">Pick a time to send</h3>
            <div class="bg-white border border-slate-200 rounded-lg p-4 flex justify-center">
              <DateTimePicker 
                id="date-picker"
                @change="handleDateChange"
              />
            </div>
             <div v-if="store.sms_scheduled_datetime" class="text-center text-sm font-medium text-purple-600">
              Scheduled for: {{ formatDateTime(store.sms_scheduled_datetime) }}
            </div>
          </div>

        </div>
        <div class="px-4 py-4 bg-slate-50 border-t border-slate-200 sm:px-6 flex justify-between">
          <button 
            id="back-sms-content"
            @click="goBack"
            class="inline-flex justify-center py-2 px-4 border border-slate-300 shadow-sm text-sm font-medium rounded-md text-slate-700 bg-white hover:bg-slate-50 focus:outline-none"
          >
            Back
          </button>
          <button 
            id="btn-schedule-sms"
            @click="scheduleCampaign"
            :disabled="!isValid"
            class="inline-flex justify-center py-2 px-6 border border-transparent shadow-sm text-sm font-bold rounded-md text-white bg-purple-600 hover:bg-purple-700 focus:outline-none disabled:opacity-50"
          >
            Schedule SMS
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import DateTimePicker from '../components/widgets/DateTimePicker.vue'

export default {
  name: 'CREATE_SMS_CAMPAIGN_REVIEW_SCHEDULE',
  components: {
    DateTimePicker
  },
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    function handleDateChange(dateStr) {
      store.sms_scheduled_datetime = dateStr
    }

    const isValid = computed(() => {
      return store.sms_scheduled_datetime && store.sms_scheduled_datetime.length > 0
    })
    
    function formatDateTime(isoStr) {
      if (!isoStr) return ''
      return new Date(isoStr).toLocaleString()
    }

    async function goBack() {
      store.setCurrentPageId('CREATE_SMS_CAMPAIGN_CONTENT')
      await router.push({ name: 'CREATE_SMS_CAMPAIGN_CONTENT' })
    }

    async function scheduleCampaign() {
      if (!isValid.value) return
      store.setCurrentPageId('SMS_CAMPAIGN_SCHEDULED_SUCCESS')
      await router.push({ name: 'SMS_CAMPAIGN_SCHEDULED_SUCCESS' })
    }

    return {
      store,
      handleDateChange,
      isValid,
      formatDateTime,
      goBack,
      scheduleCampaign
    }
  }
}
</script>