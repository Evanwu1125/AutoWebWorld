<template>
  <div class="min-h-screen bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
    <div class="max-w-2xl mx-auto">
      <div class="bg-white shadow sm:rounded-lg overflow-hidden">
        <div class="px-4 py-5 sm:p-6">
          <h3 class="text-lg leading-6 font-medium text-gray-900 text-center mb-8">
            Rate this Course
          </h3>
          
          <div class="space-y-8">
            <!-- Star Rating -->
            <div class="flex justify-center space-x-2">
               <!-- Simple 5 stars, click 5th for action -->
               <span 
                 v-for="i in 4" 
                 :key="i"
                 class="text-4xl cursor-pointer text-gray-300 hover:text-yellow-400 transition-colors"
               >★</span>
               <span 
                 id="rating-stars-5"
                 class="text-4xl cursor-pointer transition-colors"
                 :class="store.rating_stars_selected ? 'text-yellow-400' : 'text-gray-300 hover:text-yellow-400'"
                 @click="selectStars"
               >★</span>
            </div>
            <p class="text-center text-sm text-gray-500" v-if="store.rating_stars_selected">
              5 stars selected!
            </p>

            <!-- Review Text -->
            <div>
              <label for="rating-textarea" class="block text-sm font-medium text-gray-700">
                Write a review
              </label>
              <div class="mt-1">
                <textarea 
                  id="rating-textarea" 
                  rows="4" 
                  class="shadow-sm focus:ring-blue-500 focus:border-blue-500 block w-full sm:text-sm border-gray-300 rounded-md p-3 border"
                  placeholder="Share your experience..."
                  @input="handleTextInput"
                ></textarea>
              </div>
            </div>

            <!-- Buttons -->
            <div class="flex justify-end space-x-4 pt-4 border-t border-gray-200">
              <button 
                id="rating-cancel-button"
                type="button" 
                class="bg-white py-2 px-4 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                @click="cancel"
              >
                Cancel
              </button>
              <button 
                id="rating-submit-button"
                type="button" 
                class="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-700 hover:bg-blue-800 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                :disabled="!canSubmit"
                @click="submitRating"
              >
                Submit Review
              </button>
            </div>
          </div>
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
  name: 'COURSE_RATING_FORM',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const canSubmit = computed(() => store.rating_stars_selected && store.rating_text_filled && store.selected_course_id)

    function selectStars() {
      store.rating_stars_selected = true
    }

    function handleTextInput(e) {
      store.rating_text_filled = e.target.value.length > 0
    }

    async function submitRating() {
      if (canSubmit.value) {
        store.setCurrentPageId('COURSE_RATING_SUBMITTED_SUCCESS')
        await router.push({ name: 'COURSE_RATING_SUBMITTED_SUCCESS' })
      }
    }

    async function cancel() {
      store.setCurrentPageId('COURSE_HOME')
      await router.push({ name: 'COURSE_HOME', params: { id: store.selected_course_id } })
    }

    return {
      store,
      canSubmit,
      selectStars,
      handleTextInput,
      submitRating,
      cancel
    }
  }
}
</script>