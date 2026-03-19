<template>
  <div class="min-h-screen bg-gray-50 flex flex-col md:flex-row text-gray-900 font-sans">
    <!-- Summary Sidebar (Right on Desktop) -->
    <div class="md:w-1/2 bg-gray-50 p-8 md:p-12 order-1 md:order-2 border-l border-gray-200">
         <div class="max-w-lg mx-auto sticky top-12">
             <div class="flex items-center gap-4 mb-6">
                 <div class="text-2xl font-bold text-gray-900">Order Summary</div>
                 <div class="text-gray-500 text-sm">({{ cartItems.length }} items)</div>
             </div>
             
             <div class="space-y-4 mb-8">
                 <div v-for="item in cartItems" :key="item.id" class="flex items-center gap-4">
                     <div class="relative w-16 h-16 bg-white border border-gray-200 rounded-lg overflow-hidden flex-shrink-0">
                         <img :src="item.image" :alt="item.title" class="w-full h-full object-cover" />
                         <span class="absolute -top-2 -right-2 bg-gray-500 text-white text-xs font-bold w-5 h-5 rounded-full flex items-center justify-center">{{ item.quantity }}</span>
                     </div>
                     <div class="flex-1">
                         <div class="font-medium text-gray-900">{{ item.title }}</div>
                         <div class="text-sm text-gray-500">{{ item.variant_title }}</div>
                     </div>
                     <div class="font-medium text-gray-900">${{ (item.price * item.quantity).toFixed(2) }}</div>
                 </div>
             </div>
             
             <div class="border-t border-gray-200 pt-6 space-y-2">
                 <div class="flex justify-between text-gray-600">
                     <span>Subtotal</span>
                     <span>${{ subtotal.toFixed(2) }}</span>
                 </div>
                 <div class="flex justify-between text-gray-600">
                     <span>Shipping</span>
                     <span class="text-xs text-gray-500">Calculated at next step</span>
                 </div>
             </div>
             <div class="border-t border-gray-200 mt-6 pt-6 flex justify-between items-center">
                 <span class="text-xl font-bold text-gray-900">Total</span>
                 <span class="text-2xl font-bold text-gray-900">${{ subtotal.toFixed(2) }}</span>
             </div>
         </div>
    </div>

    <!-- Main Content (Left on Desktop) -->
    <div class="md:w-1/2 bg-white p-8 md:p-12 order-2 md:order-1">
        <div class="max-w-lg mx-auto">
            <h1 class="text-2xl font-bold text-[#008060] mb-8 tracking-tight">STOREFRONT</h1>
            
            <nav class="flex items-center text-xs md:text-sm text-gray-500 mb-8 font-medium">
                <span class="text-[#008060]">Information</span>
                <span class="mx-2">›</span>
                <span>Shipping</span>
                <span class="mx-2">›</span>
                <span>Payment</span>
            </nav>

            <div class="space-y-8">
                <!-- Contact Info -->
                <section>
                    <div class="flex justify-between items-center mb-4">
                        <h2 class="text-lg font-bold text-gray-900">Contact information</h2>
                        <span class="text-sm text-[#008060] cursor-pointer hover:underline" @click="goBackCart">Log in</span>
                    </div>
                    <div>
                        <input 
                            id="checkout-email"
                            type="email" 
                            v-model="email" 
                            placeholder="Email address"
                            class="w-full border-gray-300 rounded-lg shadow-sm focus:border-[#008060] focus:ring focus:ring-[#008060] focus:ring-opacity-50 py-3 px-4"
                        />
                    </div>
                </section>

                <!-- Shipping Address -->
                <section>
                    <h2 class="text-lg font-bold text-gray-900 mb-4">Shipping address</h2>
                    <div class="grid grid-cols-2 gap-4 mb-4">
                        <input 
                            id="checkout-first-name"
                            type="text" 
                            v-model="firstName" 
                            placeholder="First name"
                            class="w-full border-gray-300 rounded-lg shadow-sm focus:border-[#008060] focus:ring focus:ring-[#008060] focus:ring-opacity-50 py-3 px-4"
                        />
                        <input 
                            id="checkout-last-name"
                            type="text" 
                            v-model="lastName" 
                            placeholder="Last name"
                            class="w-full border-gray-300 rounded-lg shadow-sm focus:border-[#008060] focus:ring focus:ring-[#008060] focus:ring-opacity-50 py-3 px-4"
                        />
                    </div>
                    <div class="mb-4">
                        <input 
                            id="checkout-address1"
                            type="text" 
                            v-model="address1" 
                            placeholder="Address"
                            class="w-full border-gray-300 rounded-lg shadow-sm focus:border-[#008060] focus:ring focus:ring-[#008060] focus:ring-opacity-50 py-3 px-4"
                        />
                    </div>
                    <div class="grid grid-cols-2 gap-4">
                        <input 
                            id="checkout-city"
                            type="text" 
                            v-model="city" 
                            placeholder="City"
                            class="w-full border-gray-300 rounded-lg shadow-sm focus:border-[#008060] focus:ring focus:ring-[#008060] focus:ring-opacity-50 py-3 px-4"
                        />
                        <input 
                            id="checkout-postcode"
                            type="text" 
                            v-model="postcode" 
                            placeholder="Postal code"
                            class="w-full border-gray-300 rounded-lg shadow-sm focus:border-[#008060] focus:ring focus:ring-[#008060] focus:ring-opacity-50 py-3 px-4"
                        />
                    </div>
                </section>

                <!-- Actions -->
                <div class="flex items-center justify-between pt-6 border-t border-gray-100">
                    <span 
                        id="checkout-back-to-cart" 
                        @click="goBackCart" 
                        class="text-[#008060] cursor-pointer hover:underline text-sm font-medium flex items-center"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
                        </svg>
                        Return to cart
                    </span>
                    <button 
                        id="checkout-continue-to-shipping" 
                        @click="continueToShipping"
                        class="bg-[#008060] hover:bg-[#004C3F] text-white font-bold py-4 px-8 rounded-lg shadow-md transition-all"
                    >
                        Continue to shipping
                    </button>
                </div>
            </div>
        </div>
    </div>
  </div>
</template>

<script>
import { computed, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'CHECKOUT_INFORMATION_MAIN',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const cartItems = computed(() => signatureStore.cart_items)
    const subtotal = computed(() => signatureStore.cart_subtotal)

    // Form fields linked to signature store via watchers or direct v-model if store allowed, 
    // but better to use local refs and sync to match FSM strictness or just computed with setter.
    // FSM uses 'type' actions which update store. We simulate this with v-model directly on store for simplicity in Vue,
    // OR local refs updated on navigation.
    // Given FSM structure: user types -> updates signature.
    
    // We'll use computed with get/set to proxy store values directly
    const email = computed({
        get: () => signatureStore.email,
        set: (val) => signatureStore.email = val
    })
    const firstName = computed({
        get: () => signatureStore.shipping_first_name,
        set: (val) => signatureStore.shipping_first_name = val
    })
    const lastName = computed({
        get: () => signatureStore.shipping_last_name,
        set: (val) => signatureStore.shipping_last_name = val
    })
    const address1 = computed({
        get: () => signatureStore.shipping_address1,
        set: (val) => signatureStore.shipping_address1 = val
    })
    const city = computed({
        get: () => signatureStore.shipping_city,
        set: (val) => signatureStore.shipping_city = val
    })
    const postcode = computed({
        get: () => signatureStore.shipping_postcode,
        set: (val) => signatureStore.shipping_postcode = val
    })

    const goBackCart = async () => {
        signatureStore.currentPageId = 'CART'
        await router.push({ name: 'CART' })
    }

    const continueToShipping = async () => {
        // Validation per FSM Preconditions
        if (
            email.value && 
            firstName.value && 
            address1.value && 
            postcode.value
        ) {
            signatureStore.currentPageId = 'CHECKOUT_SHIPPING_MAIN'
            await router.push({ name: 'CHECKOUT_SHIPPING_MAIN' })
        } else {
            alert('Please fill in all required fields.')
        }
    }

    return {
        cartItems,
        subtotal,
        email,
        firstName,
        lastName,
        address1,
        city,
        postcode,
        goBackCart,
        continueToShipping
    }
  }
}
</script>