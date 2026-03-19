<template>
  <div class="min-h-screen bg-gray-50 py-12 px-4 sm:px-6 lg:px-8 font-sans">
    <div class="max-w-3xl mx-auto">
        <div class="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
            <div class="bg-[#008060] p-6 text-white text-center">
                <h1 class="text-3xl font-bold mb-2">Thank you, {{ firstName }}!</h1>
                <p class="opacity-90">Order {{ orderId }} confirmed</p>
            </div>
            
            <div class="p-8">
                <div class="border-b border-gray-100 pb-8 mb-8">
                    <p class="text-gray-600 mb-4">
                        We've sent a confirmation email to <span class="font-semibold text-gray-900">{{ email }}</span>.
                    </p>
                    <div class="bg-gray-50 rounded-lg p-4 border border-gray-100">
                        <h3 class="font-bold text-gray-900 mb-2">Shipping to:</h3>
                        <p class="text-gray-600">{{ firstName }} {{ lastName }}</p>
                        <p class="text-gray-600">{{ address1 }}</p>
                        <p class="text-gray-600">{{ city }} {{ postcode }}</p>
                    </div>
                </div>
                
                <button 
                  id="order-confirmation-go-home" 
                  @click="goHome" 
                  class="w-full bg-[#008060] hover:bg-[#004C3F] text-white font-bold py-4 px-8 rounded-lg shadow-md transition-all"
                >
                  Continue Shopping
                </button>
            </div>
        </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'ORDER_CONFIRMATION_SUCCESS',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const orderId = computed(() => signatureStore.order_id || '#12345')
    const email = computed(() => signatureStore.email || 'customer@example.com')
    const firstName = computed(() => signatureStore.shipping_first_name || 'Customer')
    const lastName = computed(() => signatureStore.shipping_last_name || '')
    const address1 = computed(() => signatureStore.shipping_address1 || '123 Main St')
    const city = computed(() => signatureStore.shipping_city || 'City')
    const postcode = computed(() => signatureStore.shipping_postcode || '12345')

    const goHome = async () => {
        signatureStore.cart_items = [] // Clear cart
        signatureStore.cart_subtotal = 0
        signatureStore.currentPageId = 'HOME'
        await router.push({ name: 'HOME' })
    }

    return {
        orderId,
        email,
        firstName,
        lastName,
        address1,
        city,
        postcode,
        goHome
    }
  }
}
</script>