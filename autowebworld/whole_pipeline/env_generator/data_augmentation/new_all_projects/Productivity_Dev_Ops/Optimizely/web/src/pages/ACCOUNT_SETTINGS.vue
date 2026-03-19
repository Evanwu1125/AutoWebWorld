<template>
  <div class="min-h-screen bg-gray-50 py-12">
    <div class="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="bg-white rounded-lg shadow-lg overflow-hidden">
        <!-- Header with Home Link -->
        <div class="px-8 py-6 border-b border-gray-200 flex justify-between items-center">
          <div>
            <h1 class="text-2xl font-bold text-gray-900">Account Settings</h1>
            <p class="mt-1 text-sm text-gray-500">Manage your profile and preferences</p>
          </div>
          <button id="logo-home" @click="goHome" class="text-blue-600 hover:text-blue-800 font-bold text-lg">Optimizely</button>
        </div>

        <!-- Tabs -->
        <div class="bg-gray-50 px-8 border-b border-gray-200">
          <nav class="-mb-px flex space-x-8">
            <button class="border-blue-500 text-blue-600 whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm">
              Profile
            </button>
            <button id="tab-billing" @click="goToBilling" class="border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm">
              Billing
            </button>
          </nav>
        </div>

        <div class="p-8 space-y-8">
          <!-- Profile Form -->
          <div>
             <label for="input-account-name" class="block text-sm font-medium text-gray-700">Account Name</label>
             <input 
               id="input-account-name"
               type="text" 
               v-model="name"
               @input="updateName"
               class="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm p-2 border"
               placeholder="Your Name"
             >
          </div>

          <!-- Notifications -->
          <div class="flex items-start">
            <div class="flex items-center h-5">
              <input 
                id="notifications-checkbox" 
                type="checkbox" 
                v-model="notifications"
                @change="updateNotifications"
                class="focus:ring-blue-500 h-4 w-4 text-blue-600 border-gray-300 rounded"
              >
            </div>
            <div class="ml-3 text-sm">
              <label for="notifications-checkbox" class="font-medium text-gray-700">Email Notifications</label>
              <p class="text-gray-500">Receive updates about your experiments.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'ACCOUNT_SETTINGS',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    
    const name = ref('')
    const notifications = ref(false)

    function updateName() {
      signatureStore.account_name = name.value
    }

    function updateNotifications() {
      signatureStore.notification_checkbox = notifications.value
    }

    function goToBilling() {
      // Only allow if name is set per precondition (length_gt 0)
      if (name.value.length > 0) {
        signatureStore.setCurrentPageId('BILLING_SETTINGS')
        router.push({ name: 'BILLING_SETTINGS' })
      } else {
        alert("Please enter an account name first.")
      }
    }

    function goHome() {
      signatureStore.setCurrentPageId('HOME')
      router.push({ name: 'HOME' })
    }

    return {
      name,
      notifications,
      updateName,
      updateNotifications,
      goToBilling,
      goHome
    }
  }
}
</script>