<template>
  <div class="min-h-screen bg-gray-50 text-gray-900 font-sans">
    <nav class="bg-white border-b border-gray-200">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <span class="text-xl font-bold text-[#008060]">Addresses</span>
            <span 
                id="addresses-back-account" 
                @click="goBackAccount"
                class="text-gray-500 hover:text-[#008060] cursor-pointer text-sm font-medium"
            >
                Back to Account
            </span>
        </div>
    </nav>

    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div class="text-center mb-12">
            <button 
                id="add-new-address" 
                @click="addNewAddress"
                class="bg-[#008060] hover:bg-[#004C3F] text-white font-bold py-3 px-6 rounded-lg shadow-md transition-colors"
            >
                Add a new address
            </button>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            <div v-for="address in addresses" :key="address.id" class="bg-white p-8 rounded-xl shadow-sm border border-gray-100 relative">
                <div v-if="address.default" class="absolute top-4 right-4 bg-gray-100 text-gray-600 text-xs font-bold px-2 py-1 rounded">DEFAULT</div>
                <h3 class="text-lg font-bold text-gray-900 mb-2">{{ address.first_name }} {{ address.last_name }}</h3>
                <div class="text-gray-600 space-y-1 mb-6">
                    <p>{{ address.address1 }}</p>
                    <p>{{ address.city }}, {{ address.postcode }}</p>
                    <p>{{ address.country }}</p>
                </div>
                <div class="flex space-x-4 text-sm font-medium">
                    <span class="text-[#008060] cursor-pointer hover:underline">Edit</span>
                    <span class="text-red-500 cursor-pointer hover:underline">Delete</span>
                </div>
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
  name: 'ACCOUNT_ADDRESSES',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const addresses = computed(() => dataStore.addresses)

    const goBackAccount = async () => {
        signatureStore.currentPageId = 'ACCOUNT_DASHBOARD'
        await router.push({ name: 'ACCOUNT_DASHBOARD' })
    }

    const addNewAddress = async () => {
        signatureStore.currentPageId = 'ADDRESS_EDIT'
        await router.push({ name: 'ADDRESS_EDIT' })
    }

    return {
        addresses,
        goBackAccount,
        addNewAddress
    }
  }
}
</script>