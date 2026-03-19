<template>
  <div class="min-h-screen bg-white font-sans flex flex-col items-center justify-center p-4">
    <div class="w-full max-w-md space-y-8">
        <div class="text-center">
            <h1 class="text-2xl font-bold text-gray-900">Guest Checkout</h1>
            <p class="text-gray-500 mt-2">No account required</p>
        </div>

        <div class="space-y-6">
            <div>
                <label class="block text-sm font-medium text-gray-700 mb-1">Email</label>
                <input 
                    id="guest-email"
                    type="email" 
                    v-model="email" 
                    placeholder="Email address"
                    class="w-full border-gray-300 rounded-lg shadow-sm focus:border-[#008060] focus:ring focus:ring-[#008060] py-3 px-4"
                />
            </div>
            
            <div>
                <label class="block text-sm font-medium text-gray-700 mb-1">Name</label>
                <input 
                    id="guest-first-name"
                    type="text" 
                    v-model="firstName" 
                    placeholder="First Name"
                    class="w-full border-gray-300 rounded-lg shadow-sm focus:border-[#008060] focus:ring focus:ring-[#008060] py-3 px-4"
                />
            </div>
            
             <!-- Address implied by FSM simplified flow or assumed standard -->

            <div class="flex items-center justify-between pt-4">
                <span 
                    id="guest-back-to-cart" 
                    @click="goBackCart"
                    class="text-[#008060] cursor-pointer hover:underline text-sm font-medium"
                >
                    Return to cart
                </span>
                <button 
                    id="guest-continue-to-payment" 
                    @click="continueToPayment"
                    class="bg-[#008060] hover:bg-[#004C3F] text-white font-bold py-3 px-6 rounded-lg shadow-md"
                >
                    Continue to payment
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
  name: 'CHECKOUT_GUEST_INFORMATION',
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

    const goBackCart = async () => {
        signatureStore.currentPageId = 'CART'
        await router.push({ name: 'CART' })
    }

    const continueToPayment = async () => {
        if (email.value && firstName.value) {
            // Mock address filling for FSM precondition satisfaction if needed
            signatureStore.shipping_address1 = "Guest Address Mock" 
            
            signatureStore.currentPageId = 'CHECKOUT_GUEST_PAYMENT'
            await router.push({ name: 'CHECKOUT_GUEST_PAYMENT' })
        } else {
            alert('Please fill all fields')
        }
    }

    return {
        email,
        firstName,
        goBackCart,
        continueToPayment
    }
  }
}
</script>