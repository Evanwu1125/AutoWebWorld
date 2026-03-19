<template>
  <div class="min-h-screen bg-slate-50 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
    <div class="max-w-md w-full text-center space-y-8">
      <div class="mx-auto flex items-center justify-center h-24 w-24 rounded-full bg-emerald-100 mb-8">
        <svg class="h-12 w-12 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
      </div>
      
      <h2 class="text-3xl font-extrabold text-slate-900">Campaign Scheduled!</h2>
      <p class="text-lg text-slate-600">Your email campaign is ready to go.</p>

      <div class="flex flex-col space-y-4 pt-8">
        <button 
          id="btn-view-campaigns"
          @click="viewCampaigns"
          class="w-full flex justify-center py-3 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none"
        >
          View All Campaigns
        </button>
        <button 
          id="btn-go-home"
          @click="goHome"
          class="w-full flex justify-center py-3 px-4 border border-slate-300 rounded-md shadow-sm text-sm font-medium text-slate-700 bg-white hover:bg-slate-50 focus:outline-none"
        >
          Return Home
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'EMAIL_CAMPAIGN_SCHEDULED_SUCCESS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    async function goHome() {
      // Effect from FSM
      store.success_message = "Email campaign scheduled"
      store.setCurrentPageId('HOME')
      await router.push({ name: 'HOME' })
    }

    async function viewCampaigns() {
      // FSM does not specify effect for this action, but let's check
      // ACT_EMAIL_SUCCESS_BACK_CAMPAIGNS has empty effects.
      store.setCurrentPageId('CAMPAIGNS_LIST')
      await router.push({ name: 'CAMPAIGNS_LIST' })
    }

    return {
      goHome,
      viewCampaigns
    }
  }
}
</script>