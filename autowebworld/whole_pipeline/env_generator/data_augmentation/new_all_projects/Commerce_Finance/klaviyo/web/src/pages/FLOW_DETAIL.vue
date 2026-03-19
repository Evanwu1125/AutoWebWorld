<template>
  <div class="min-h-screen bg-slate-50">
    <header class="bg-white border-b border-slate-200">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <button 
          id="back-flows"
          @click="goBack"
          class="text-sm font-medium text-slate-500 hover:text-blue-600 flex items-center"
        >
          <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
          Back to Flows
        </button>
      </div>
    </header>

    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8" v-if="flow">
      <div class="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
        <div class="p-6 border-b border-slate-100 flex items-center space-x-4">
           <div class="w-16 h-16 rounded bg-emerald-100 flex items-center justify-center shrink-0">
               <img :src="flow.image" class="w-full h-full object-cover rounded opacity-80" />
           </div>
           <div>
             <h1 class="text-2xl font-bold text-slate-900">{{ flow.name }}</h1>
             <p class="text-slate-500">Trigger: {{ flow.trigger }}</p>
           </div>
           <div class="ml-auto">
             <span class="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium capitalize"
              :class="{
                'bg-emerald-100 text-emerald-800': flow.status === 'live',
                'bg-slate-100 text-slate-800': flow.status === 'draft'
              }">
              {{ flow.status }}
            </span>
           </div>
        </div>

        <div class="p-8">
           <h3 class="text-lg font-bold mb-4">Flow Performance</h3>
           <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div class="bg-slate-50 p-6 rounded-xl text-center border border-slate-100">
                <div class="text-3xl font-bold text-slate-900">${{ flow.revenue.toLocaleString() }}</div>
                <div class="text-sm text-slate-500 uppercase tracking-wide mt-1">Total Revenue</div>
              </div>
           </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { computed, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'FLOW_DETAIL',
  setup() {
    const router = useRouter()
    const route = useRoute()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    onMounted(() => {
      if (!store.selected_flow_id && route.params.id) {
        store.selected_flow_id = route.params.id
      }
    })

    const flow = computed(() => {
      return dataStore.flows.find(f => f.id === store.selected_flow_id)
    })

    async function goBack() {
      store.setCurrentPageId('FLOWS_LIST')
      await router.push({ name: 'FLOWS_LIST' })
    }

    return {
      flow,
      goBack
    }
  }
}
</script>