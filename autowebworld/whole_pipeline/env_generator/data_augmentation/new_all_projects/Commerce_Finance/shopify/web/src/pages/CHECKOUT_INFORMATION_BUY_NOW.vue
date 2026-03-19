<template>
  <div class="min-h-screen bg-white font-sans flex flex-col items-center justify-center p-4">
    <div class="w-full max-w-md space-y-8">
        <div class="text-center">
            <h1 class="text-2xl font-bold text-gray-900">Buy Now Checkout</h1>
            <p class="text-gray-500 mt-2">Express purchase for {{ signatureStore.selected_product_id }}</p>
        </div>

        <div class="space-y-6">
            <div>
                <label class="block text-sm font-medium text-gray-700 mb-1">Email</label>
                <input 
                    id="checkout-email-buy-now"
                    type="email" 
                    v-model="email" 
                    placeholder="Email address"
                    class="w-full border-gray-300 rounded-lg shadow-sm focus:border-[#008060] focus:ring focus:ring-[#008060] py-3 px-4"
                />
            </div>
            
            <div>
                <label class="block text-sm font-medium text-gray-700 mb-1">Name</label>
                <input 
                    id="checkout-first-name-buy-now"
                    type="text" 
                    v-model="firstName" 
                    placeholder="First Name"
                    class="w-full border-gray-300 rounded-lg shadow-sm focus:border-[#008060] focus:ring focus:ring-[#008060] py-3 px-4"
                />
            </div>

            <!-- Note: FSM only asks for Name/Address in simplified form here based on actions provided -->

            <div class="flex space-x-4 pt-4">
                 <button 
                    id="checkout-buy-now-back-product" 
                    @click="goBackProduct"
                    class="flex-1 bg-white border border-gray-300 text-gray-700 font-bold py-3 px-4 rounded-lg hover:bg-gray-50"
                >
                    Cancel
                </button>
                <button 
                    id="checkout-continue-buy-now" 
                    @click="continueToPayment"
                    class="flex-1 bg-[#008060] hover:bg-[#004C3F] text-white font-bold py-3 px-4 rounded-lg shadow-md"
                >
                    Payment
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
  name: 'CHECKOUT_INFORMATION_BUY_NOW',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const email = computed({
        get: () => signatureStore.email,
        set: (val) => signatureStore.email = val
    })
    const firstName = computed({
        get: () => signatureStore.shipping_first_name,
        set: (val) => signatureStore.shipping_first_name = val
    })

    const goBackProduct = async () => {
        // Need to know which product we came from, stored in signature
        const prodId = signatureStore.selected_product_id
        signatureStore.currentPageId = 'PRODUCT_DETAIL'
        await router.push({ name: 'PRODUCT_DETAIL', params: { id: prodId } })
    }

    const continueToPayment = async () => {
        if (email.value && firstName.value) {
            signatureStore.currentPageId = 'CHECKOUT_PAYMENT_BUY_NOW'
            await router.push({ name: 'CHECKOUT_PAYMENT_BUY_NOW' })
        } else {
            alert('Please fill required fields.')
        }
    }

    return {
        signatureStore,
        email,
        firstName,
        goBackProduct,
        continueToPayment
    }
  }
}
</script>