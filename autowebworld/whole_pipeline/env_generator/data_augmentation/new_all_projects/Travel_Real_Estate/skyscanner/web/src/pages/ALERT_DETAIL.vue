<template>
  <div class="min-h-screen bg-slate-50 font-sans pb-12">
    <header class="bg-white shadow-sm sticky top-0 z-30">
      <div class="max-w-2xl mx-auto px-6 h-16 flex items-center justify-between">
        <div id="back-alerts-list" @click="goBack" class="flex items-center gap-2 cursor-pointer text-[#002D5C] hover:text-blue-600 transition-colors">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"/></svg>
          <span class="font-medium">Back to Alerts</span>
        </div>
        <div class="font-bold text-[#002D5C]">Alert Details</div>
        <div class="w-24"></div>
      </div>
    </header>

    <main class="max-w-2xl mx-auto px-6 py-8 space-y-6" v-if="alert">
      <div class="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
        <div class="h-48 relative">
           <img :src="alert.image" class="w-full h-full object-cover" />
           <div class="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent"></div>
           <div class="absolute bottom-6 left-6 text-white">
             <h1 class="text-3xl font-bold">{{ alert.origin }} → {{ alert.destination }}</h1>
             <div class="flex items-center gap-2 mt-1">
               <div :class="{'bg-green-500': alert.active, 'bg-gray-500': !alert.active}" class="w-2 h-2 rounded-full"></div>
               <span class="text-sm font-medium">{{ alert.active ? 'Tracking Active' : 'Tracking Paused' }}</span>
             </div>
           </div>
        </div>
        
        <div class="p-8 grid grid-cols-2 gap-8">
           <div>
             <div class="text-xs font-bold text-gray-400 uppercase tracking-wide mb-1">Target Price</div>
             <div class="text-2xl font-bold text-gray-900">£{{ alert.target_price }}</div>
           </div>
           <div>
             <div class="text-xs font-bold text-gray-400 uppercase tracking-wide mb-1">Current Best</div>
             <div class="text-2xl font-bold text-[#0770E3]">£{{ alert.current_price }}</div>
           </div>
        </div>
      </div>

      <div class="bg-white rounded-2xl shadow-sm border border-gray-100 p-8 text-center space-y-4">
        <h3 class="font-bold text-gray-900">Manage Alert</h3>
        <p class="text-gray-500 text-sm">Update preferences or delete this alert.</p>
        <button class="w-full py-3 border border-gray-200 hover:bg-gray-50 rounded-xl font-medium text-gray-700 transition-colors">
          Edit Parameters
        </button>
        <button class="w-full py-3 border border-red-100 text-red-600 hover:bg-red-50 rounded-xl font-medium transition-colors">
          Delete Alert
        </button>
      </div>
      
      <div class="text-center pt-8">
        <button id="alert-detail-home" @click="goHome" class="text-sm text-gray-500 hover:text-blue-600 underline">
          Go to Home
        </button>
      </div>

    </main>
  </div>
</template>

<script>
import { ref, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'ALERT_DETAIL',
  setup() {
    const router = useRouter()
    const route = useRoute()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const alert = ref(null)

    onMounted(() => {
      const id = route.params.id || store.alert_selected_id
      alert.value = dataStore.priceAlerts.find(a => a.id === id)
    })

    const goBack = async () => {
      store.currentPageId = 'PRICE_ALERTS_LIST'
      await router.push({ name: 'PRICE_ALERTS_LIST' })
    }

    const goHome = async () => {
      store.currentPageId = 'HOME'
      await router.push({ name: 'HOME' })
    }

    return {
      alert,
      goBack,
      goHome
    }
  }
}
</script>