<template>
  <div class="min-h-screen bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
    <div class="max-w-xl mx-auto">
      <h2 class="text-3xl font-extrabold text-gray-900 text-center mb-8">Review Order</h2>
      
      <div class="bg-white shadow overflow-hidden sm:rounded-lg mb-6">
        <div class="px-4 py-5 sm:px-6 border-b border-gray-200">
          <h3 class="text-lg leading-6 font-medium text-gray-900">Order Summary</h3>
        </div>
        
        <div class="px-4 py-5 sm:p-6">
          <div v-if="course" class="flex items-start mb-6">
            <img :src="course.image" alt="Course" class="h-16 w-16 object-cover rounded-md mr-4">
            <div>
              <h4 class="text-lg font-bold text-gray-900">{{ course.title }}</h4>
              <p class="text-sm text-gray-500">{{ course.university }}</p>
            </div>
          </div>

          <div class="border-t border-gray-200 pt-4 space-y-2">
            <div class="flex justify-between text-sm text-gray-600">
              <span>Subtotal</span>
              <span>$49.99</span>
            </div>
            <div class="flex justify-between text-sm text-gray-600">
              <span>Tax</span>
              <span>$0.00</span>
            </div>
            <div class="flex justify-between text-lg font-bold text-gray-900 pt-2 border-t border-gray-200 mt-2">
              <span>Total</span>
              <span>$49.99</span>
            </div>
          </div>
        </div>

        <div class="bg-gray-50 px-4 py-4 sm:px-6 flex justify-between items-center">
          <div 
            id="order-review-back-button"
            @click="goBack"
            class="text-sm font-medium text-blue-600 hover:text-blue-500 cursor-pointer"
          >
            Back to Payment
          </div>

          <button 
            id="place-order-button"
            type="button" 
            class="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-blue-700 hover:bg-blue-800 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            @click="placeOrder"
          >
            Pay Now
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
import { useDataStore } from '../stores/data'

export default {
  name: 'ORDER_REVIEW',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const course = computed(() => dataStore.courses.find(c => c.id === store.selected_course_id))

    async function placeOrder() {
      // Add to enrolled courses mock
      dataStore.enrolled_courses.push({
        ...course.value,
        enrolled_as: 'purchase'
      })

      store.setCurrentPageId('ENROLL_COURSE_SUCCESS')
      await router.push({ name: 'ENROLL_COURSE_SUCCESS' })
    }

    async function goBack() {
      store.setCurrentPageId('PAYMENT_DETAILS')
      await router.push({ name: 'PAYMENT_DETAILS' })
    }

    return {
      course,
      placeOrder,
      goBack
    }
  }
}
</script>