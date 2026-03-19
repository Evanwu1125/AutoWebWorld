<template>
  <div class="min-h-screen bg-white text-gray-900 font-sans p-8">
    <div class="max-w-2xl mx-auto">
        <h1 class="text-3xl font-bold mb-8 text-center">Order Summary</h1>
        
        <div class="bg-gray-50 p-8 rounded-xl border border-gray-200 mb-8 space-y-4" v-if="order">
            <div class="flex justify-between border-b border-gray-200 pb-4">
                <span class="text-gray-600">Order Number</span>
                <span class="font-bold">{{ order.order_number }}</span>
            </div>
             <div class="flex justify-between border-b border-gray-200 pb-4">
                <span class="text-gray-600">Date</span>
                <span class="font-bold">{{ order.date }}</span>
            </div>
             <div class="flex justify-between border-b border-gray-200 pb-4">
                <span class="text-gray-600">Status</span>
                <span class="font-bold">{{ order.financial_status }} / {{ order.fulfillment_status }}</span>
            </div>
            
             <div class="pt-4">
                <h3 class="font-bold mb-2">Shipping Address</h3>
                <p class="text-gray-600">{{ order.shipping_address.address1 }}</p>
                <p class="text-gray-600">{{ order.shipping_address.city }} {{ order.shipping_address.postcode }}</p>
            </div>
        </div>

        <div class="flex space-x-4">
             <button 
                id="summary-back-to-order" 
                @click="goBackOrder"
                class="flex-1 bg-white border border-gray-300 hover:bg-gray-50 text-gray-700 font-bold py-3 px-6 rounded-lg transition-colors"
            >
                Back to Order
            </button>
             <button 
                id="summary-back-account" 
                @click="goBackAccount"
                class="flex-1 bg-[#008060] hover:bg-[#004C3F] text-white font-bold py-3 px-6 rounded-lg transition-colors"
            >
                Back to Account
            </button>
        </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'ORDER_SUMMARY',
  setup() {
    const router = useRouter()
    const route = useRoute()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const order = computed(() => dataStore.orders.find(o => o.id === route.params.id))

    const goBackOrder = async () => {
        signatureStore.currentPageId = 'ORDER_DETAIL'
        await router.push({ name: 'ORDER_DETAIL', params: { id: route.params.id } })
    }

    const goBackAccount = async () => {
        signatureStore.currentPageId = 'ACCOUNT_DASHBOARD'
        await router.push({ name: 'ACCOUNT_DASHBOARD' })
    }

    return {
        order,
        goBackOrder,
        goBackAccount
    }
  }
}
</script>