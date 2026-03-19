<template>
  <div class="min-h-screen bg-slate-50 py-12">
    <div class="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="bg-white shadow rounded-lg overflow-hidden">
        <div class="px-4 py-5 sm:p-6">
          <h2 class="text-2xl font-bold text-slate-900 mb-6">Review & Schedule</h2>
          
          <div class="bg-blue-50 rounded-lg p-4 mb-8 border border-blue-100">
            <h4 class="font-bold text-blue-900 mb-2">Campaign Summary</h4>
            <ul class="text-sm text-blue-800 space-y-1">
              <li><strong>Name:</strong> {{ store.campaign_name }}</li>
              <li><strong>Subject:</strong> {{ store.subject_line }}</li>
              <li><strong>List:</strong> {{ store.selected_list_id }}</li>
            </ul>
          </div>

          <div class="space-y-4">
            <h3 class="text-lg font-medium text-slate-900">Pick a time to send</h3>
            <div class="bg-white border border-slate-200 rounded-lg p-4 flex justify-center">
              <!-- Reusing existing DateTimePicker widget -->
              <DateTimePicker 
                id="date-picker"
                @change="handleDateChange"
              />
            </div>
            
            <div v-if="store.email_scheduled_datetime" class="text-center text-sm font-medium text-emerald-600">
              Scheduled for: {{ formatDateTime(store.email_scheduled_datetime) }}
            </div>
          </div>

        </div>
        <div class="px-4 py-4 bg-slate-50 border-t border-slate-200 sm:px-6 flex justify-between">
          <button 
            id="back-content"
            @click="goBack"
            class="inline-flex justify-center py-2 px-4 border border-slate-300 shadow-sm text-sm font-medium rounded-md text-slate-700 bg-white hover:bg-slate-50 focus:outline-none"
          >
            Back
          </button>
          <button 
            id="btn-schedule-campaign"
            @click="scheduleCampaign"
            :disabled="!isValid"
            class="inline-flex justify-center py-2 px-6 border border-transparent shadow-sm text-sm font-bold rounded-md text-white bg-emerald-600 hover:bg-emerald-700 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Schedule Campaign
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
  name: 'CREATE_CAMPAIGN_REVIEW_SCHEDULE',
  components: {
    DateTimePicker
  },
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    function handleDateChange(dateStr) {
      // dateStr format expected to be ISO string or standard format from widget
      store.email_scheduled_datetime = dateStr
    }

    const isValid = computed(() => {
      // Precondition: length_gt 0
      return store.email_scheduled_datetime && store.email_scheduled_datetime.length > 0
    })

    function formatDateTime(isoStr) {
      if (!isoStr) return ''
      return new Date(isoStr).toLocaleString()
    }

    async function goBack() {
      store.setCurrentPageId('CREATE_CAMPAIGN_CONTENT')
      await router.push({ name: 'CREATE_CAMPAIGN_CONTENT' })
    }

    async function scheduleCampaign() {
      if (!isValid.value) return
      store.setCurrentPageId('EMAIL_CAMPAIGN_SCHEDULED_SUCCESS')
      await router.push({ name: 'EMAIL_CAMPAIGN_SCHEDULED_SUCCESS' })
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