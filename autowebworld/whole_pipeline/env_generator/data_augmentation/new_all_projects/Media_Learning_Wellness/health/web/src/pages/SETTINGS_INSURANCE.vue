<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <header class="bg-white shadow-sm z-10">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
        <h1 class="text-2xl font-bold text-[#005DAA]">Insurance Settings</h1>
        <button id="back-settings-account" @click="handleBack" class="text-gray-600 hover:text-gray-900">
          Back to Profile
        </button>
      </div>
    </header>

    <main class="flex-1 max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
      <div class="bg-white rounded-lg shadow mb-8 p-6">
        <h2 class="text-lg font-bold text-gray-900 mb-6">Update Insurance Details</h2>

        <div class="space-y-6">
           <div>
              <label for="insurance-member-id" class="block text-sm font-medium text-gray-700 mb-2">Member ID</label>
              <input
                id="insurance-member-id"
                type="text"
                class="shadow-sm focus:ring-[#009CDE] focus:border-[#009CDE] block w-full sm:text-sm border-gray-300 rounded-md py-2 px-3"
                placeholder="Enter ID from card"
                :value="store.insurance_member_id_entered"
                @input="handleIdInput"
              />
           </div>

           <div class="pt-4 border-t border-gray-200 flex justify-end">
              <button
                id="insurance-save"
                @click="handleSave"
                class="bg-[#005DAA] text-white py-2 px-6 rounded-md font-bold hover:bg-[#004a87] shadow-sm transition-colors"
              >
                Save Insurance
              </button>
           </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'SETTINGS_INSURANCE',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const handleIdInput = (e) => {
      // ACT_INS_TYPE_MEMBER_ID
      store.insurance_member_id_entered = e.target.value
    }

    const handleSave = () => {
      // ACT_INS_SAVE
      if (store.insurance_member_id_entered.length > 0) {
        alert('Insurance Info Saved!')
      } else {
        alert('Member ID is required.')
      }
    }

    const handleBack = async () => {
      // ACT_INS_BACK_ACCOUNT
      store.setCurrentPageId('SETTINGS_ACCOUNT')
      await router.push({ name: 'SETTINGS_ACCOUNT' })
    }

    return {
      store,
      handleIdInput,
      handleSave,
      handleBack
    }
  }
}
</script>