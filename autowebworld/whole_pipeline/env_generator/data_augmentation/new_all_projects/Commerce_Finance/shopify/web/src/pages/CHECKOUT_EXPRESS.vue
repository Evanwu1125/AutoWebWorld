<template>
  <div class="min-h-screen bg-[#F6F6F7] font-sans flex items-center justify-center p-4">
    <div class="w-full max-w-md bg-white p-8 rounded-xl shadow-md">
        <div class="text-center mb-8">
            <h1 class="text-2xl font-bold text-gray-900">Express Checkout</h1>
            <p class="text-gray-500">Fast & Secure</p>
        </div>

        <div class="space-y-6">
            <div>
                <label class="block text-sm font-medium text-gray-700 mb-2">Select Provider</label>
                <div class="relative" id="express-provider-dropdown">
                    <select 
                        v-model="provider" 
                        class="w-full border-gray-300 rounded-lg shadow-sm focus:border-[#008060] focus:ring focus:ring-[#008060] py-3 px-4 appearance-none"
                    >
                        <option value="" disabled>Choose provider</option>
                        <option value="shop_pay" id="express-provider-shop-pay">Shop Pay</option>
                        <option value="paypal" id="express-provider-paypal">PayPal</option>
                        <option value="google_pay" id="express-provider-google-pay">Google Pay</option>
                    </select>
                </div>
            </div>

            <button 
                id="express-pay-now" 
                @click="payNow"
                :disabled="!provider"
                class="w-full bg-[#5A31F4] hover:bg-[#4825C9] text-white font-bold py-4 px-6 rounded-lg shadow-md transition-colors flex items-center justify-center disabled:opacity-50"
            >
                Pay with {{ providerName }}
            </button>
            
            <button 
                id="express-back-to-cart" 
                @click="goBackCart"
                class="w-full text-gray-500 hover:text-gray-900 text-sm font-medium"
            >
                Cancel and return to cart
            </button>
        </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'CHECKOUT_EXPRESS',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const provider = computed({
        get: () => signatureStore.express_provider,
        set: (val) => signatureStore.express_provider = val
    })

    const providerName = computed(() => {
        if (provider.value === 'shop_pay') return 'Shop Pay'
        if (provider.value === 'paypal') return 'PayPal'
        if (provider.value === 'google_pay') return 'Google Pay'
        return 'Provider'
    })

    const goBackCart = async () => {
        signatureStore.currentPageId = 'CART'
        await router.push({ name: 'CART' })
    }

    const payNow = async () => {
        if (provider.value) {
            signatureStore.order_id = `EXP-${Math.floor(Math.random() * 10000)}`
            signatureStore.currentPageId = 'CHECKOUT_SUCCESS_EXPRESS'
            await router.push({ name: 'CHECKOUT_SUCCESS_EXPRESS' })
        }
    }

    return {
        provider,
        providerName,
        goBackCart,
        payNow
    }
  }
}
</script>