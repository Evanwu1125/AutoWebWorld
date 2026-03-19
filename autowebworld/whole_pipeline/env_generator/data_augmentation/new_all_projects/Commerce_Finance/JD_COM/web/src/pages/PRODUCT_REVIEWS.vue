<template>
  <div class="min-h-screen bg-[#F6F6F6]">
    <!-- Header -->
    <header class="bg-white shadow-sm sticky top-0 z-20">
      <div class="container mx-auto px-4 py-3 flex items-center gap-4">
        <button id="back-product" @click="goBack" class="text-gray-600 hover:text-[#E1251B] flex items-center gap-1">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
          Back to Product
        </button>
        <h1 class="text-lg font-bold">Write a Review</h1>
      </div>
    </header>

    <main class="container mx-auto px-4 py-6 max-w-2xl">
      <div class="bg-white rounded-xl shadow-sm p-6">
        <h2 class="text-xl font-bold mb-6">Rate this product</h2>
        
        <!-- Star Rating -->
        <div class="flex gap-2 mb-8">
          <div 
            v-for="star in 5" 
            :key="star"
            :id="`rating-star-${star}`"
            @click="setRating(star)"
            class="text-4xl cursor-pointer transition-transform hover:scale-110"
            :class="star <= (currentRating || 0) ? 'text-yellow-400' : 'text-gray-300'"
          >
            ★
          </div>
        </div>

        <!-- Review Text -->
        <div class="mb-6">
          <label class="block text-sm font-medium text-gray-700 mb-2">Your Review</label>
          <textarea 
            id="review-textarea"
            v-model="reviewText"
            @input="handleInput"
            rows="6"
            class="w-full border border-gray-300 rounded-lg p-4 outline-none focus:border-[#E1251B] focus:ring-1 focus:ring-[#E1251B] resize-none"
            placeholder="Share your experience with this product..."
          ></textarea>
        </div>

        <!-- Submit -->
        <button 
          id="btn-submit-review"
          @click="submitReview"
          class="w-full bg-[#E1251B] text-white font-bold py-3 rounded-lg hover:bg-[#c91f16] transition-colors shadow-lg shadow-red-200 disabled:opacity-50 disabled:cursor-not-allowed"
          :disabled="!canSubmit"
        >
          Submit Review
        </button>
      </div>
    </main>
  </div>
</template>

<script>
import { ref, computed } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'PRODUCT_REVIEWS',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();

    const currentRating = computed(() => signatureStore.review_rating);
    const reviewText = ref('');

    const canSubmit = computed(() => {
      return currentRating.value > 0 && signatureStore.review_text_entered === true;
    });

    const setRating = (star) => {
      signatureStore.review_rating = star; // Only 5 is mapped in FSM but logic allows any
    };

    const handleInput = () => {
      if (reviewText.value.length > 0) {
        signatureStore.review_text_entered = true;
      }
    };

    const submitReview = async () => {
      signatureStore.currentPageId = 'REVIEW_SUBMITTED_SUCCESS';
      await router.push({ name: 'REVIEW_SUBMITTED_SUCCESS' });
    };

    const goBack = async () => {
      signatureStore.currentPageId = 'PRODUCT_DETAIL';
      await router.push({ name: 'PRODUCT_DETAIL' });
    };

    return {
      currentRating,
      reviewText,
      canSubmit,
      setRating,
      handleInput,
      submitReview,
      goBack
    };
  }
}
</script>