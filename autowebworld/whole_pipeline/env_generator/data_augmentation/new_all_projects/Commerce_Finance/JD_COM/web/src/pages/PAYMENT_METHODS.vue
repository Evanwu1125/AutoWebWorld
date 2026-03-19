<template>
  <div class="min-h-screen bg-[#F6F6F6]">
    <header class="bg-white shadow-sm sticky top-0 z-20">
      <div class="container mx-auto px-4 py-4 flex items-center gap-4">
        <button id="back-user-center" @click="goBack" class="text-gray-600 hover:text-[#E1251B] flex items-center gap-1">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
          Back
        </button>
        <h1 class="text-xl font-bold">Payment Methods</h1>
      </div>
    </header>

    <main class="container mx-auto px-4 py-6 max-w-2xl">
      <!-- Existing Cards -->
      <div class="space-y-4 mb-8">
        <div v-for="card in cards" :key="card.id" class="bg-gradient-to-r from-gray-700 to-gray-900 p-6 rounded-xl shadow-lg text-white relative overflow-hidden">
          <div class="absolute -right-6 -top-6 w-24 h-24 bg-white/10 rounded-full"></div>
          <div class="flex justify-between items-start mb-8">
            <span class="font-bold text-lg tracking-wider">{{ card.type.toUpperCase() }}</span>
            <span class="text-2xl">💳</span>
          </div>
          <div class="font-mono text-xl tracking-widest mb-4">**** **** **** {{ card.last4 }}</div>
          <div class="text-xs opacity-70 uppercase">Card Holder</div>
          <div class="font-bold tracking-wide">{{ card.cardHolder }}</div>
        </div>
      </div>

      <!-- Add New Card -->
      <div class="bg-white rounded-xl shadow-md p-6">
        <h2 class="text-lg font-bold mb-4">Add New Card</h2>
        <div class="space-y-4">
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Card Number</label>
            <input 
              id="card-number"
              type="text"
              v-model="cardNumber"
              @input="handleNumber"
              class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:border-[#E1251B] outline-none"
              placeholder="16-digit number"
            />
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Card Holder Name</label>
            <input 
              id="card-holder"
              type="text"
              v-model="cardHolder"
              @input="handleHolder"
              class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:border-[#E1251B] outline-none"
              placeholder="Name on card"
            />
          </div>
          <button 
            id="btn-save-card"
            @click="save"
            class="w-full bg-[#E1251B] text-white font-bold py-3 rounded-lg shadow hover:bg-[#c91f16] disabled:opacity-50"
            :disabled="!canSave"
          >
            Save Card
          </button>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { ref, computed } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';

export default {
  name: 'PAYMENT_METHODS',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const cardNumber = ref('');
    const cardHolder = ref('');
    const cards = computed(() => dataStore.paymentMethods);

    const numEntered = computed(() => signatureStore.payment_card_number_entered);
    const holderEntered = computed(() => signatureStore.payment_card_holder_entered);
    const canSave = computed(() => numEntered.value && holderEntered.value);

    const handleNumber = () => {
      if (cardNumber.value.length > 0) signatureStore.payment_card_number_entered = true;
    };

    const handleHolder = () => {
      if (cardHolder.value.length > 0) signatureStore.payment_card_holder_entered = true;
    };

    const save = async () => {
      signatureStore.currentPageId = 'USER_CENTER';
      await router.push({ name: 'USER_CENTER' });
    };

    const goBack = async () => {
      signatureStore.currentPageId = 'USER_CENTER';
      await router.push({ name: 'USER_CENTER' });
    };

    return {
      cardNumber,
      cardHolder,
      cards,
      canSave,
      handleNumber,
      handleHolder,
      save,
      goBack
    };
  }
}
</script>