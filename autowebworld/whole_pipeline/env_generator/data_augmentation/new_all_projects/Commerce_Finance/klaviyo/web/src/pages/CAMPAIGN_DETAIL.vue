<template>
  <div class="min-h-screen bg-slate-50">
    <header class="bg-white border-b border-slate-200">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <button 
          id="back-campaigns-list"
          @click="goBack"
          class="text-sm font-medium text-slate-500 hover:text-blue-600 flex items-center"
        >
          <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
          Back to Campaigns
        </button>
        
        <button 
          id="btn-duplicate-campaign"
          @click="duplicateCampaign"
          class="bg-white border border-slate-300 text-slate-700 hover:bg-slate-50 font-medium py-2 px-4 rounded-lg shadow-sm"
        >
          Duplicate Campaign
        </button>
      </div>
    </header>

    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8" v-if="campaign">
      <div class="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
        <div class="h-48 relative">
           <img :src="campaign.image" class="w-full h-full object-cover" />
           <div class="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent"></div>
           <div class="absolute bottom-4 left-6 text-white">
             <h1 class="text-3xl font-bold">{{ campaign.name }}</h1>
             <p class="opacity-90">{{ campaign.type.toUpperCase() }} • {{ campaign.status.toUpperCase() }}</p>
           </div>
        </div>
        
        <div class="p-6 grid grid-cols-1 md:grid-cols-3 gap-8">
          <div class="bg-slate-50 p-4 rounded-lg border border-slate-100">
            <h3 class="text-xs font-bold text-slate-500 uppercase tracking-wide mb-1">Total Revenue</h3>
            <p class="text-3xl font-bold text-slate-900">${{ campaign.revenue.toLocaleString() }}</p>
          </div>
          
           <div class="bg-slate-50 p-4 rounded-lg border border-slate-100">
            <h3 class="text-xs font-bold text-slate-500 uppercase tracking-wide mb-1">Sent Date</h3>
            <p class="text-2xl font-semibold text-slate-800">{{ campaign.sent || 'Not sent yet' }}</p>
          </div>
          
           <div class="bg-slate-50 p-4 rounded-lg border border-slate-100">
            <h3 class="text-xs font-bold text-slate-500 uppercase tracking-wide mb-1">ID</h3>
            <p class="text-lg font-mono text-slate-600">{{ campaign.id }}</p>
          </div>
        </div>
      </div>
    </main>
    
    <div v-else class="max-w-7xl mx-auto px-4 py-8 text-center text-slate-500">
      Campaign not found.
    </div>
  </div>
</template>

<script>
import { computed, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'CAMPAIGN_DETAIL',
  setup() {
    const router = useRouter()
    const route = useRoute()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    // Ensure selected_campaign_id is set if refreshed directly
    onMounted(() => {
      if (!store.selected_campaign_id && route.params.id) {
        store.selected_campaign_id = route.params.id
      }
    })

    const campaign = computed(() => {
      return dataStore.campaigns.find(c => c.id === store.selected_campaign_id)
    })

    async function goBack() {
      store.setCurrentPageId('CAMPAIGNS_LIST')
      await router.push({ name: 'CAMPAIGNS_LIST' })
    }

    async function duplicateCampaign() {
      if (!store.selected_campaign_id) return
      store.setCurrentPageId('CREATE_CAMPAIGN_BASICS')
      await router.push({ name: 'CREATE_CAMPAIGN_BASICS' })
    }

    return {
      campaign,
      goBack,
      duplicateCampaign
    }
  }
}
</script>