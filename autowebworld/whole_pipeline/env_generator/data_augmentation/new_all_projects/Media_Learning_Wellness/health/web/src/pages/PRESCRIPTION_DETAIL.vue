<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <!-- Header -->
    <header class="bg-white shadow-sm z-10">
       <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center">
         <button id="back-rx-list" @click="handleBack" class="mr-4 text-gray-600 hover:text-gray-900">
           <svg class="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>
         </button>
         <h1 class="text-xl font-bold text-gray-900">Prescription Details</h1>
       </div>
    </header>

    <main class="flex-1 max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
      <div class="bg-white rounded-lg shadow-lg overflow-hidden">
        <div class="p-6 border-b border-gray-200 flex justify-between items-start">
           <div>
              <h2 class="text-2xl font-bold text-gray-900">{{ prescription?.name }}</h2>
              <p class="text-gray-500">{{ prescription?.dosage }}</p>
           </div>
           <span 
             class="px-3 py-1 text-sm font-semibold rounded-full"
             :class="prescription?.status === 'Active' ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'"
           >
             {{ prescription?.status }}
           </span>
        </div>

        <div class="p-6 space-y-6">
           <div class="grid grid-cols-2 gap-4">
              <div>
                <p class="text-sm font-medium text-gray-500">Supply</p>
                <p class="text-base text-gray-900">{{ prescription?.supply }}</p>
              </div>
              <div>
                 <p class="text-sm font-medium text-gray-500">Refills Remaining</p>
                 <p class="text-base text-gray-900">2</p>
              </div>
           </div>

           <div>
              <label for="renewal-notes" class="block text-sm font-medium text-gray-700 mb-2">
                Renewal Notes (Optional)
              </label>
              <textarea
                id="renewal-notes"
                rows="3"
                class="shadow-sm focus:ring-[#009CDE] focus:border-[#009CDE] block w-full sm:text-sm border-gray-300 rounded-md"
                placeholder="Add instructions for the pharmacist..."
                @input="handleNotesInput"
              ></textarea>
           </div>
        </div>

        <div class="p-6 bg-gray-50">
           <button
             id="continue-renewal"
             @click="handleRenew"
             class="w-full bg-[#005DAA] text-white py-3 px-4 rounded-lg font-bold hover:bg-[#004a87] shadow-md transition-colors"
           >
             Request Renewal
           </button>
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
  name: 'PRESCRIPTION_DETAIL',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const prescription = computed(() => {
      return dataStore.prescriptions.find(r => r.id === store.selected_prescription_id)
    })

    const handleNotesInput = (e) => {
      // ACT_RX_DETAIL_TYPE_NOTES
      store.renewal_notes = e.target.value
    }

    const handleRenew = async () => {
      // ACT_RX_DETAIL_CONTINUE
      // Precondition: selected_prescription_id exists (checked by computed/routing context, technically length > 0)
      if (store.selected_prescription_id) {
        store.setCurrentPageId('PRESCRIPTION_RENEWAL_REVIEW')
        await router.push({ name: 'PRESCRIPTION_RENEWAL_REVIEW' })
      }
    }

    const handleBack = async () => {
      // ACT_RX_DETAIL_BACK_LIST
      store.setCurrentPageId('PRESCRIPTION_LIST')
      await router.push({ name: 'PRESCRIPTION_LIST' })
    }

    return {
      prescription,
      handleNotesInput,
      handleRenew,
      handleBack
    }
  }
}
</script>