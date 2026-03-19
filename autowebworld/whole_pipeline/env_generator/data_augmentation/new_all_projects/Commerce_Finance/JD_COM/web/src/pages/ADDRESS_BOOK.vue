<template>
  <div class="min-h-screen bg-[#F6F6F6]">
    <header class="bg-white shadow-sm sticky top-0 z-20">
      <div class="container mx-auto px-4 py-4 flex items-center gap-4">
        <button id="back-user-center" @click="goBack" class="text-gray-600 hover:text-[#E1251B] flex items-center gap-1">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
          Back
        </button>
        <h1 class="text-xl font-bold">Address Book</h1>
      </div>
    </header>

    <main class="container mx-auto px-4 py-6 max-w-2xl">
      <!-- Existing Addresses -->
      <div class="space-y-4 mb-8">
        <div v-for="addr in addresses" :key="addr.id" class="bg-white p-4 rounded-xl shadow-sm border border-gray-100 flex justify-between items-center">
          <div>
            <div class="font-bold text-gray-900">{{ addr.name }} <span v-if="addr.isDefault" class="bg-red-100 text-[#E1251B] text-xs px-2 py-0.5 rounded ml-2">Default</span></div>
            <div class="text-gray-600 text-sm mt-1">{{ addr.detail }}</div>
          </div>
          <button class="text-gray-400 hover:text-gray-600">Edit</button>
        </div>
      </div>

      <!-- Add New Form -->
      <div class="bg-white rounded-xl shadow-md p-6">
        <h2 class="text-lg font-bold mb-4">Add New Address</h2>
        <div class="space-y-4">
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Recipient Name</label>
            <input 
              id="address-name"
              type="text"
              v-model="name"
              @input="handleName"
              class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:border-[#E1251B] outline-none"
            />
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Detailed Address</label>
            <input 
              id="address-detail"
              type="text"
              v-model="detail"
              @input="handleDetail"
              class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:border-[#E1251B] outline-none"
            />
          </div>
          <button 
            id="btn-save-address"
            @click="save"
            class="w-full bg-[#E1251B] text-white font-bold py-3 rounded-lg shadow hover:bg-[#c91f16] disabled:opacity-50"
            :disabled="!canSave"
          >
            Save Address
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
  name: 'ADDRESS_BOOK',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const name = ref('');
    const detail = ref('');
    const addresses = computed(() => dataStore.addresses);

    const nameEntered = computed(() => signatureStore.address_form_name_entered);
    const detailEntered = computed(() => signatureStore.address_form_detail_entered);
    const canSave = computed(() => nameEntered.value && detailEntered.value);

    const handleName = () => {
      if (name.value.length > 0) signatureStore.address_form_name_entered = true;
    };

    const handleDetail = () => {
      if (detail.value.length > 0) signatureStore.address_form_detail_entered = true;
    };

    const save = async () => {
      // Logic to save would go here
      signatureStore.currentPageId = 'USER_CENTER';
      await router.push({ name: 'USER_CENTER' });
    };

    const goBack = async () => {
      signatureStore.currentPageId = 'USER_CENTER';
      await router.push({ name: 'USER_CENTER' });
    };

    return {
      name,
      detail,
      addresses,
      canSave,
      handleName,
      handleDetail,
      save,
      goBack
    };
  }
}
</script>