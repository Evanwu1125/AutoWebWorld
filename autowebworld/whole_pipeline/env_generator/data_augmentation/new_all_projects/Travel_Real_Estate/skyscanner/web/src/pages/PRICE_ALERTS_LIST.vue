<template>
  <div class="min-h-screen bg-slate-50 font-sans">
    <header class="bg-[#002D5C] text-white py-4 px-6 shadow-md sticky top-0 z-30">
      <div class="max-w-4xl mx-auto flex items-center justify-between">
        <div id="back-account-overview" @click="goBack" class="flex items-center gap-2 cursor-pointer hover:text-blue-200 transition-colors">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"/></svg>
          <span class="font-medium">My Account</span>
        </div>
        <div class="font-bold text-xl">Price Alerts</div>
        <div class="w-24"></div>
      </div>
    </header>

    <main class="max-w-4xl mx-auto px-6 py-8 space-y-6">
      <div class="bg-white rounded-xl shadow-sm p-6 border border-gray-100 flex items-center justify-between">
         <span class="font-bold text-gray-700">Filters</span>
         <label class="flex items-center gap-3 cursor-pointer group">
            <div class="relative flex items-center">
              <input 
                id="alerts-filter-active-checkbox"
                type="checkbox" 
                @change="handleFilterActive"
                class="peer h-5 w-5 cursor-pointer appearance-none rounded border border-gray-300 shadow-sm transition-all checked:border-blue-600 checked:bg-blue-600 hover:border-blue-400 focus:ring-2 focus:ring-blue-200" 
              />
              <svg class="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-3.5 h-3.5 pointer-events-none opacity-0 peer-checked:opacity-100 text-white" viewBox="0 0 14 14" fill="none">
                <path d="M3 8L6 11L11 3.5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
              </svg>
            </div>
            <span class="text-gray-700 group-hover:text-blue-600 transition-colors">Active Alerts Only</span>
          </label>
      </div>

      <div id="alerts-list" @scroll="handleScroll" class="space-y-4">
        <div 
          v-for="alert in filteredAlerts" 
          :key="alert.id"
          :class="[
            `data-id-${alert.id}`,
            'bg-white rounded-2xl shadow-sm hover:shadow-md transition-all border border-gray-100 p-6 flex items-center gap-6 cursor-pointer group',
            isFiltered ? 'alert-row-filtered' : 'alert-row-visible'
          ]"
          @click="openAlert(alert.id)"
        >
          <img :src="alert.image" class="w-16 h-16 rounded-xl object-cover" />
          
          <div class="flex-1">
            <h3 class="font-bold text-gray-900 group-hover:text-blue-600 transition-colors">{{ alert.origin }} to {{ alert.destination }}</h3>
            <p class="text-sm text-gray-500">Target: £{{ alert.target_price }}</p>
          </div>

          <div class="text-right">
             <div class="text-xl font-bold text-gray-900">£{{ alert.current_price }}</div>
             <div :class="{'text-green-500': alert.active, 'text-gray-400': !alert.active}" class="text-xs font-bold uppercase tracking-wide">
               {{ alert.active ? 'Active' : 'Paused' }}
             </div>
          </div>
          
           <svg class="w-5 h-5 text-gray-300 group-hover:text-blue-500 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/></svg>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'PRICE_ALERTS_LIST',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const isFiltered = computed(() => store.alerts_list_filters_applied)

    const filteredAlerts = computed(() => {
      let result = [...dataStore.priceAlerts]
      // Since FSM action `ALERTS_LIST_FILTER_ACTIVE` sets `alerts_list_filters_applied`, 
      // we assume it means filtering by active status.
      if (store.alerts_list_filters_applied) {
        result = result.filter(a => a.active)
      }
      return result
    })

    const handleFilterActive = () => {
      store.alerts_list_filters_applied = true
    }

    const handleScroll = (e) => {
      if (filteredAlerts.value.length > 0) {
        store.alerts_list_viewport_anchor_id = filteredAlerts.value[0].id
      }
    }

    const openAlert = async (id) => {
      store.alert_selected_id = id
      
      // Clear flags
      if (store.alerts_list_filters_applied) store.alerts_list_filters_applied = null
      if (store.alerts_list_viewport_anchor_id) store.alerts_list_viewport_anchor_id = null

      store.currentPageId = 'ALERT_DETAIL'
      await router.push({ name: 'ALERT_DETAIL', params: { id } })
    }

    const goBack = async () => {
      store.currentPageId = 'ACCOUNT_OVERVIEW'
      await router.push({ name: 'ACCOUNT_OVERVIEW' })
    }

    return {
      filteredAlerts,
      isFiltered,
      handleFilterActive,
      handleScroll,
      openAlert,
      goBack
    }
  }
}
</script>