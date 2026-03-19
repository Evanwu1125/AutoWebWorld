<template>
  <div class="checkout-page min-h-screen bg-gray-50 flex flex-col">
    <!-- Simplified Checkout Header -->
    <header class="bg-white border-b p-4 sticky top-0 z-30">
      <div class="max-w-3xl mx-auto flex justify-center">
         <div class="font-bold text-xl text-[#0071DC] flex items-center gap-2">
            <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" /></svg>
            Walmart Checkout
         </div>
      </div>
    </header>

    <main class="flex-1 max-w-3xl mx-auto w-full p-4 md:p-8">
      <div class="bg-white rounded-xl shadow-sm overflow-hidden">
        <!-- Progress Steps -->
        <div class="flex border-b">
           <div class="flex-1 py-3 text-center text-blue-600 font-bold border-b-2 border-blue-600">1. Shipping</div>
           <div class="flex-1 py-3 text-center text-gray-400 font-medium">2. Payment</div>
           <div class="flex-1 py-3 text-center text-gray-400 font-medium">3. Review</div>
        </div>

        <div class="p-6 md:p-8 space-y-6">
           <h2 class="text-2xl font-bold mb-6">Where should we send your order?</h2>
           
           <!-- Form -->
           <div class="space-y-4">
              <div class="form-group">
                 <label class="block text-sm font-medium text-gray-700 mb-1">Full Name</label>
                 <input 
                   id="shipping-name-input"
                   type="text" 
                   v-model="fullName"
                   @input="updateFullName"
                   class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none transition-shadow"
                   placeholder="First and Last Name"
                 />
              </div>
              <div class="form-group">
                 <label class="block text-sm font-medium text-gray-700 mb-1">Address</label>
                 <input 
                   id="shipping-address-line1-input"
                   type="text" 
                   v-model="address"
                   @input="updateAddress"
                   class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none transition-shadow"
                   placeholder="Street address"
                 />
              </div>
              <div class="grid grid-cols-2 gap-4">
                 <div class="form-group">
                   <label class="block text-sm font-medium text-gray-700 mb-1">City</label>
                   <input 
                     id="shipping-city-input"
                     type="text" 
                     v-model="city"
                     @input="updateCity"
                     class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none transition-shadow"
                     placeholder="City"
                   />
                 </div>
                 <div class="form-group">
                   <label class="block text-sm font-medium text-gray-700 mb-1">ZIP Code</label>
                   <input 
                     id="shipping-zip-input"
                     type="text" 
                     v-model="zip"
                     @input="updateZip"
                     class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none transition-shadow"
                     placeholder="ZIP Code"
                   />
                 </div>
              </div>

              <!-- Shipping Method -->
              <div class="form-group pt-4">
                <label class="block text-sm font-medium text-gray-700 mb-2">Delivery Options</label>
                <div class="relative">
                   <button 
                     id="shipping-method-dropdown"
                     @click="showMethodDropdown = !showMethodDropdown"
                     class="w-full flex items-center justify-between px-4 py-3 border border-gray-300 rounded-lg bg-white text-left hover:border-gray-400"
                   >
                     <span>{{ methodLabel }}</span>
                     <svg class="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" /></svg>
                   </button>
                   
                   <div v-if="showMethodDropdown" class="absolute z-10 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg">
                      <div 
                        id="shipping-method-standard"
                        @click="selectMethod('delivery')"
                        class="px-4 py-3 hover:bg-gray-50 cursor-pointer flex justify-between items-center"
                      >
                         <div>
                           <div class="font-medium">Standard Delivery</div>
                           <div class="text-xs text-gray-500">3-5 Business Days</div>
                         </div>
                         <span v-if="method === 'delivery'" class="text-blue-600 font-bold">✓</span>
                      </div>
                      <div 
                        id="shipping-method-pickup"
                        @click="selectMethod('pickup')"
                        class="px-4 py-3 hover:bg-gray-50 cursor-pointer flex justify-between items-center border-t border-gray-100"
                      >
                         <div>
                           <div class="font-medium">Store Pickup</div>
                           <div class="text-xs text-gray-500">Ready in 2 hours</div>
                         </div>
                         <span v-if="method === 'pickup'" class="text-blue-600 font-bold">✓</span>
                      </div>
                   </div>
                </div>
              </div>
           </div>
           
           <!-- Actions -->
           <div class="pt-6 border-t flex items-center justify-between">
              <button 
                id="shipping-back-to-cart"
                @click="handleBackToCart"
                class="text-gray-600 font-medium hover:text-[#0071DC] hover:underline"
              >
                &larr; Back to Cart
              </button>
              <button 
                id="shipping-continue-button"
                @click="handleContinue"
                :disabled="!isFormValid"
                class="bg-[#0071DC] text-white font-bold py-3 px-8 rounded-full shadow-md hover:bg-[#005bb5] disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                Continue to Payment
              </button>
           </div>

        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'CHECKOUT_SHIPPING',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const fullName = ref(store.shipping_full_name || '')
    const address = ref(store.shipping_address_line1 || '')
    const city = ref(store.shipping_city || '')
    const zip = ref(store.shipping_zip || '')
    const method = ref(store.shipping_method || 'delivery')
    const showMethodDropdown = ref(false)

    const isFormValid = computed(() => {
      return fullName.value && address.value && city.value && zip.value && method.value
    })

    const methodLabel = computed(() => {
      return method.value === 'delivery' ? 'Standard Delivery' : 'Store Pickup'
    })

    const updateFullName = () => { store.shipping_full_name = fullName.value } // FSM: ACT_SHIPPING_ENTER_NAME
    const updateAddress = () => { store.shipping_address_line1 = address.value } // FSM: ACT_SHIPPING_ENTER_ADDRESS
    const updateCity = () => { store.shipping_city = city.value } // FSM: ACT_SHIPPING_ENTER_CITY
    const updateZip = () => { store.shipping_zip = zip.value } // FSM: ACT_SHIPPING_ENTER_ZIP
    
    const selectMethod = (val) => { 
      // FSM: ACT_SHIPPING_SELECT_METHOD
      method.value = val
      store.shipping_method = val
      showMethodDropdown.value = false
    }

    const handleContinue = async () => {
      // FSM: ACT_SHIPPING_CONTINUE_TO_PAYMENT
      store.currentPageId = 'CHECKOUT_PAYMENT'
      await router.push({ name: 'CHECKOUT_PAYMENT' })
    }

    const handleBackToCart = async () => {
      // FSM: ACT_SHIPPING_BACK_TO_CART
      store.currentPageId = 'CART'
      await router.push({ name: 'CART' })
    }

    return {
      fullName, address, city, zip, method, showMethodDropdown,
      methodLabel, isFormValid,
      updateFullName, updateAddress, updateCity, updateZip, selectMethod,
      handleContinue, handleBackToCart
    }
  }
}
</script>