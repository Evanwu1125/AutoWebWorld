<template>
  <div class="flex h-screen bg-[#121212] text-white font-sans overflow-hidden">
    <aside class="w-64 bg-black flex-shrink-0 p-6 border-r border-[#282828] hidden md:block">
      <div id="back-account-overview" @click="handleBackAccount" class="flex items-center space-x-2 text-[#B3B3B3] hover:text-white cursor-pointer font-bold mb-8">
         <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/></svg>
         <span>Back to Account</span>
      </div>
    </aside>

    <main class="flex-1 overflow-y-auto p-8 md:p-12 max-w-4xl mx-auto w-full">
      <h1 class="text-3xl font-bold mb-8">Payment Methods</h1>

      <!-- Filters -->
      <div class="mb-6 flex space-x-4">
         <div id="payment-filter-active" @click="handleFilterCheckbox" class="px-4 py-2 bg-[#282828] hover:bg-[#3E3E3E] rounded-full cursor-pointer transition-colors border border-transparent" :class="{'bg-white text-black': filtersApplied}">
            <span class="font-bold text-sm">Active Only</span>
         </div>
      </div>

      <!-- Methods List -->
      <div 
        id="payment-methods" 
        class="space-y-4"
      >
        <div v-if="filteredMethods.length === 0" class="text-[#B3B3B3]">No payment methods found.</div>
        
        <div 
           v-for="method in filteredMethods" 
           :key="method.id"
           class="bg-[#181818] p-6 rounded-lg flex items-center justify-between cursor-pointer hover:bg-[#282828] transition-colors group"
           :class="[
              `data-id-${method.id}`,
              filtersApplied ? 'method-row-filtered' : 'method-row-visible'
           ]"
           @click="filtersApplied ? handleOpenFilteredMethod(method) : handleOpenMethod(method)"
        >
           <div class="flex items-center space-x-4">
              <div class="bg-[#333] p-2 rounded text-white font-mono font-bold">{{ method.type }}</div>
              <div>
                 <div class="font-bold text-white">{{ method.name }}</div>
                 <div class="text-[#B3B3B3] text-sm">Ending in •••• {{ method.last4 }}</div>
              </div>
           </div>
           <div class="text-right">
              <div class="text-sm font-bold" :class="method.is_active ? 'text-[#1DB954]' : 'text-[#B3B3B3]'">
                 {{ method.is_active ? 'Active' : 'Inactive' }}
              </div>
              <div class="text-xs text-[#727272]">Exp: {{ method.expiry }}</div>
           </div>
           
           <svg class="w-6 h-6 text-[#B3B3B3] group-hover:text-white opacity-0 group-hover:opacity-100 transition-opacity" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/></svg>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'
import { useRouter } from 'vue-router'

export default {
  name: 'PAYMENT_METHODS',
  setup() {
    const store = useSignatureStore()
    const dataStore = useDataStore()
    const router = useRouter()

    const filtersApplied = ref(false)

    const filteredMethods = computed(() => {
       if (filtersApplied.value) {
          return dataStore.payment_methods.filter(m => m.is_active)
       }
       return dataStore.payment_methods
    })

    const handleBackAccount = async () => {
       store.setCurrentPageId('ACCOUNT_OVERVIEW')
       await router.push({ name: 'ACCOUNT_OVERVIEW' })
    }

    const handleFilterCheckbox = () => {
       filtersApplied.value = true
       store.payment_methods_filters_applied = true
    }

    const handleOpenFilteredMethod = async (method) => {
       store.selected_payment_method_id = method.id
       store.payment_methods_filters_applied = null
       store.setCurrentPageId('PAYMENT_METHOD_DETAIL')
       await router.push({ name: 'PAYMENT_METHOD_DETAIL', params: { id: method.id } })
    }

    const handleOpenMethod = async (method) => {
       store.selected_payment_method_id = method.id
       store.payment_methods_viewport_anchor_id = null
       store.setCurrentPageId('PAYMENT_METHOD_DETAIL')
       await router.push({ name: 'PAYMENT_METHOD_DETAIL', params: { id: method.id } })
    }

    return {
       filtersApplied,
       filteredMethods,
       handleBackAccount,
       handleFilterCheckbox,
       handleOpenFilteredMethod,
       handleOpenMethod
    }
  }
}
</script>