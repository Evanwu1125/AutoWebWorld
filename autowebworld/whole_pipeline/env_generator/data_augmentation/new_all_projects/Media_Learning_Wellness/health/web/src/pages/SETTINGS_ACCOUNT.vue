<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <header class="bg-white shadow-sm z-10">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
        <h1 class="text-2xl font-bold text-[#005DAA]">Account Settings</h1>
        <button id="back-dashboard" @click="handleBack" class="text-gray-600 hover:text-gray-900">
          Done
        </button>
      </div>
    </header>

    <main class="flex-1 max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
      <div class="bg-white rounded-lg shadow mb-8">
        <div class="border-b border-gray-200">
          <nav class="-mb-px flex">
            <a href="#" class="border-[#005DAA] text-[#005DAA] whitespace-nowrap py-4 px-8 border-b-2 font-medium text-sm">
              Profile
            </a>
            <a 
              id="settings-insurance-tab"
              @click="handleToInsurance"
              class="border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 whitespace-nowrap py-4 px-8 border-b-2 font-medium text-sm cursor-pointer"
            >
              Insurance
            </a>
          </nav>
        </div>

        <div class="p-6 space-y-6">
           <div>
              <label for="settings-full-name" class="block text-sm font-medium text-gray-700 mb-2">Full Name</label>
              <input
                id="settings-full-name"
                type="text"
                class="shadow-sm focus:ring-[#009CDE] focus:border-[#009CDE] block w-full sm:text-sm border-gray-300 rounded-md py-2 px-3"
                placeholder="John Doe"
                :value="store.full_name_entered"
                @input="handleNameInput"
              />
           </div>

           <div>
              <label for="settings-phone" class="block text-sm font-medium text-gray-700 mb-2">Phone Number</label>
              <input
                id="settings-phone"
                type="tel"
                class="shadow-sm focus:ring-[#009CDE] focus:border-[#009CDE] block w-full sm:text-sm border-gray-300 rounded-md py-2 px-3"
                placeholder="555-123-4567"
                :value="store.phone_number_entered"
                @input="handlePhoneInput"
              />
           </div>

           <div class="pt-4 border-t border-gray-200 flex justify-end">
              <button
                id="settings-save"
                @click="handleSave"
                class="bg-[#005DAA] text-white py-2 px-6 rounded-md font-bold hover:bg-[#004a87] shadow-sm transition-colors"
              >
                Save Changes
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
  name: 'SETTINGS_ACCOUNT',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const handleNameInput = (e) => {
      // ACT_SETTINGS_TYPE_NAME
      store.full_name_entered = e.target.value
    }

    const handlePhoneInput = (e) => {
      // ACT_SETTINGS_TYPE_PHONE
      store.phone_number_entered = e.target.value
    }

    const handleSave = () => {
      // ACT_SETTINGS_SAVE
      // Precondition: full_name_entered > 0
      if (store.full_name_entered.length > 0) {
        alert('Settings Saved!')
      } else {
        alert('Name is required.')
      }
    }

    const handleToInsurance = async () => {
      // ACT_SETTINGS_GO_INSURANCE
      store.setCurrentPageId('SETTINGS_INSURANCE')
      await router.push({ name: 'SETTINGS_INSURANCE' })
    }

    const handleBack = async () => {
      // ACT_SETTINGS_BACK_DASH
      store.setCurrentPageId('DASHBOARD')
      await router.push({ name: 'DASHBOARD' })
    }

    return {
      store,
      handleNameInput,
      handlePhoneInput,
      handleSave,
      handleToInsurance,
      handleBack
    }
  }
}
</script>