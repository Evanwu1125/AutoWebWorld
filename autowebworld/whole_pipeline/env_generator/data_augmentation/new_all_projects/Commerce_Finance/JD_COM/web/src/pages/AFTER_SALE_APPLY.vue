<template>
  <div class="min-h-screen bg-[#F6F6F6]">
    <header class="bg-white shadow-sm sticky top-0 z-20">
      <div class="container mx-auto px-4 py-4 flex items-center gap-4">
        <button id="back-order-detail" @click="goBack" class="text-gray-600 hover:text-[#E1251B] flex items-center gap-1">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
          Back
        </button>
        <h1 class="text-xl font-bold">Apply for Service</h1>
      </div>
    </header>

    <main class="container mx-auto px-4 py-6 max-w-2xl">
      <div class="bg-white rounded-xl shadow-sm p-6">
        <!-- Service Type -->
        <div class="mb-6">
          <h2 class="text-lg font-bold mb-3">Service Type</h2>
          <div class="flex gap-4">
            <div class="flex-1 border-2 border-[#E1251B] bg-red-50 text-[#E1251B] rounded-lg p-4 text-center font-bold cursor-pointer">
              Return / Refund
            </div>
            <div class="flex-1 border border-gray-200 rounded-lg p-4 text-center text-gray-600 cursor-not-allowed opacity-50">
              Exchange
            </div>
          </div>
        </div>

        <!-- Reason -->
        <div class="mb-6">
          <h2 class="text-lg font-bold mb-3">Reason for Application</h2>
          <div 
            class="reason-option-quality border rounded-lg p-3 cursor-pointer hover:border-red-200 transition-colors mb-2"
            :class="reasonSelected ? 'border-[#E1251B] bg-red-50 text-[#E1251B]' : 'border-gray-200'"
            @click="selectReason"
          >
            Quality Issue
          </div>
          <div class="border border-gray-200 rounded-lg p-3 text-gray-500">Other</div>
        </div>

        <!-- Description -->
        <div class="mb-8">
          <h2 class="text-lg font-bold mb-3">Problem Description</h2>
          <textarea 
            id="after-sale-description"
            v-model="description"
            @input="handleInput"
            rows="4"
            class="w-full border border-gray-300 rounded-lg p-3 outline-none focus:border-[#E1251B]"
            placeholder="Please describe the issue in detail..."
          ></textarea>
        </div>

        <button 
          id="btn-submit-after-sale"
          @click="submit"
          class="w-full bg-[#E1251B] text-white font-bold py-3 rounded-xl shadow-lg shadow-red-200 disabled:opacity-50 disabled:cursor-not-allowed"
          :disabled="!canSubmit"
        >
          Submit Application
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
  name: 'AFTER_SALE_APPLY',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();

    const description = ref('');
    
    const reasonSelected = computed(() => signatureStore.after_sale_reason_selected);
    const descEntered = computed(() => signatureStore.after_sale_description_entered);

    const canSubmit = computed(() => reasonSelected.value && descEntered.value);

    const selectReason = () => {
      signatureStore.after_sale_reason_selected = true;
    };

    const handleInput = () => {
      if (description.value.length > 0) {
        signatureStore.after_sale_description_entered = true;
      }
    };

    const submit = async () => {
      signatureStore.currentPageId = 'ORDER_SUBMITTED_SUCCESS';
      await router.push({ name: 'ORDER_SUBMITTED_SUCCESS' });
    };

    const goBack = async () => {
      signatureStore.currentPageId = 'ORDER_DETAIL';
      await router.push({ name: 'ORDER_DETAIL' });
    };

    return {
      description,
      reasonSelected,
      canSubmit,
      selectReason,
      handleInput,
      submit,
      goBack
    };
  }
}
</script>