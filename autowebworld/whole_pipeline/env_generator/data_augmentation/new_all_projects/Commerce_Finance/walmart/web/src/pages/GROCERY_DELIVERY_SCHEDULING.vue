<template>
  <div class="grocery-scheduling-page min-h-screen bg-gray-50 flex flex-col">
    <header class="bg-[#2A8703] text-white p-4 sticky top-0 z-30 shadow-md">
      <div class="max-w-7xl mx-auto flex items-center justify-between">
         <div class="font-bold text-xl flex items-center gap-2">
            <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" /></svg>
            Walmart Grocery
         </div>
         <h1 class="text-lg font-medium">Schedule Delivery</h1>
      </div>
    </header>

    <main class="flex-1 max-w-3xl mx-auto w-full p-4 md:p-8">
      <div class="bg-white rounded-xl shadow-sm overflow-hidden">
        <div class="flex border-b">
           <div class="flex-1 py-3 text-center text-[#2A8703] font-bold border-b-2 border-[#2A8703]">1. Schedule</div>
           <div class="flex-1 py-3 text-center text-gray-400 font-medium">2. Review</div>
        </div>

        <div class="p-6 md:p-8">
           <h2 class="text-2xl font-bold mb-6 text-gray-900">Choose a delivery time</h2>
           
           <div class="mb-8">
             <label class="block text-sm font-semibold text-gray-700 mb-2">Select Date & Time</label>
             <DateTimePicker 
               id="date-picker"
               v-model="selectedDate"
               @update:modelValue="handleDateSelect"
             />
             <p class="text-sm text-gray-500 mt-2">Available slots shown in local time.</p>
           </div>
           
           <div v-if="slot" class="p-4 bg-green-50 border border-green-100 rounded-lg flex items-center gap-3 mb-8">
              <svg class="w-6 h-6 text-[#2A8703]" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
              <div>
                <div class="font-bold text-[#2A8703]">Selected Slot</div>
                <div>{{ slot }}</div>
              </div>
           </div>

           <!-- Actions -->
           <div class="pt-6 border-t flex items-center justify-between">
              <button 
                id="grocery-scheduling-back-to-cart"
                @click="handleBackToCart"
                class="text-gray-600 font-medium hover:text-[#2A8703] hover:underline"
              >
                &larr; Back to Basket
              </button>
              <button 
                id="grocery-scheduling-continue-button"
                @click="handleContinue"
                :disabled="!slot"
                class="bg-[#2A8703] text-white font-bold py-3 px-8 rounded-full shadow-md hover:bg-[#237002] disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                Continue to Review
              </button>
           </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import DateTimePicker from '../components/widgets/DateTimePicker.vue'

export default {
  name: 'GROCERY_DELIVERY_SCHEDULING',
  components: {
    DateTimePicker
  },
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const selectedDate = ref('') // DateTimePicker v-model
    const slot = ref(store.grocery_delivery_slot || '')

    const handleDateSelect = (val) => {
      // FSM: ACT_GROCERY_SELECT_SLOT_DATE
      // Val format from component: YYYY-MM-DD HH:mm
      // FSM sets a value like "2025-10-22-morning" but we'll use the actual date string for display/logic
      // However, FSM effect hardcodes value "2025-10-22-morning" in the example effect.
      // But typically we should use the selected value. 
      // The FSM example shows clicking specific elements. 
      // Our DateTimePicker updates a string.
      // We will update the store with the selected string for visual feedback, 
      // and ensure the store state is populated.
      slot.value = val
      store.grocery_delivery_slot = val
    }

    const handleContinue = async () => {
      // FSM: ACT_GROCERY_CONTINUE_TO_REVIEW
      store.currentPageId = 'GROCERY_CHECKOUT_REVIEW'
      await router.push({ name: 'GROCERY_CHECKOUT_REVIEW' })
    }

    const handleBackToCart = async () => {
      // FSM: ACT_GROCERY_SCHEDULING_BACK_TO_CART
      store.currentPageId = 'GROCERY_CART'
      await router.push({ name: 'GROCERY_CART' })
    }

    return {
      selectedDate,
      slot,
      handleDateSelect,
      handleContinue,
      handleBackToCart
    }
  }
}
</script>