<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <header class="bg-white shadow-sm z-10">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
        <h1 class="text-2xl font-bold text-[#005DAA]">Your Benefits</h1>
        <button id="back-home" @click="handleBack" class="text-gray-600 hover:text-gray-900">
          Home
        </button>
      </div>
    </header>

    <main class="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
      <div class="mb-6 flex items-center">
         <input
           id="filter-eligible-checkbox"
           type="checkbox"
           v-model="filterEligible"
           @change="handleFilterChange"
           class="h-4 w-4 text-[#005DAA] focus:ring-[#005DAA] border-gray-300 rounded"
         />
         <label for="filter-eligible-checkbox" class="ml-2 block text-sm text-gray-900">
           Show Eligible Only
         </label>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div 
          v-for="plan in filteredPlans" 
          :key="plan.id"
          class="bg-white rounded-lg shadow-lg overflow-hidden flex flex-col"
        >
          <img :src="plan.image" class="h-40 w-full object-cover" />
          <div class="p-6 flex-1 flex flex-col">
             <div class="flex justify-between items-start mb-4">
                <h3 class="text-xl font-bold text-gray-900">{{ plan.name }}</h3>
                <span 
                   class="px-2 py-1 text-xs font-semibold rounded-full"
                   :class="plan.eligible ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'"
                >
                   {{ plan.eligible ? 'Eligible' : 'Not Eligible' }}
                </span>
             </div>
             <p class="text-gray-600 mb-4 flex-1">Type: {{ plan.type }}</p>
             
             <button
               id="benefit-plan-filtered"
               @click="handleSelectPlan"
               class="w-full bg-white border border-[#005DAA] text-[#005DAA] py-2 px-4 rounded-md hover:bg-blue-50 font-medium transition-colors"
               v-if="filterEligible && plan.eligible"
             >
               View Details
             </button>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'BENEFITS_OVERVIEW',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const filterEligible = ref(false)

    const filteredPlans = computed(() => {
      let result = dataStore.plans
      if (filterEligible.value) {
        result = result.filter(p => p.eligible)
      }
      return result
    })

    const handleFilterChange = () => {
      // ACT_BENEFITS_FILTER_ELIGIBLE
      store.benefits_list_filters_applied = true
    }

    const handleSelectPlan = async () => {
      // ACT_BENEFITS_OPEN_FILTERED_PLAN
      // Precondition: filter applied
      if (store.benefits_list_filters_applied) {
        store.benefits_list_filters_applied = false
        store.setCurrentPageId('SETTINGS_INSURANCE')
        await router.push({ name: 'SETTINGS_INSURANCE' })
      }
    }

    const handleBack = async () => {
      // ACT_BENEFITS_BACK_HOME
      store.setCurrentPageId('HOME')
      await router.push({ name: 'HOME' })
    }

    return {
      filterEligible,
      filteredPlans,
      handleFilterChange,
      handleSelectPlan,
      handleBack
    }
  }
}
</script>