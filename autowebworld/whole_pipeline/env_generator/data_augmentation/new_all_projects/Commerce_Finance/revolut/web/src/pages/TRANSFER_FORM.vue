<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <!-- Top Nav -->
    <div class="bg-white px-4 py-4 shadow-sm sticky top-0 z-20 flex items-center justify-between">
      <button 
        id="back-transfer-form" 
        @click="goBack"
        class="p-2 -ml-2 rounded-full hover:bg-gray-100 text-gray-600"
      >
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
      </button>
      <h1 class="text-lg font-bold text-gray-900">New Transfer</h1>
      <div class="w-10"></div>
    </div>

    <div class="flex-1 p-6 max-w-lg mx-auto w-full">
      
      <!-- From Account Selection -->
      <div class="mb-6 relative">
        <label class="block text-sm font-medium text-gray-700 mb-2">From</label>
        <div 
          id="from-account-dropdown"
          @click="showAccountDropdown = !showAccountDropdown"
          class="w-full bg-white border border-gray-300 rounded-xl p-4 flex items-center justify-between cursor-pointer hover:border-blue-500 transition-colors shadow-sm"
        >
          <div v-if="selectedAccount" class="flex items-center gap-3">
            <img :src="selectedAccount.image" class="w-8 h-8 rounded-full object-cover" />
            <div>
              <div class="font-bold text-gray-900">{{ selectedAccount.name }}</div>
              <div class="text-xs text-gray-500">{{ selectedAccount.balance }} {{ selectedAccount.currency }}</div>
            </div>
          </div>
          <span v-else class="text-gray-400">Select account</span>
          <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
        </div>

        <!-- Dropdown Options -->
        <div v-if="showAccountDropdown" class="absolute top-full left-0 w-full mt-2 bg-white rounded-xl shadow-xl border border-gray-100 z-50 max-h-60 overflow-y-auto">
          <div 
            v-for="(acc, index) in accounts" 
            :key="acc.id"
            :id="`from-account-option-${index + 1}`"
            :class="['p-4 hover:bg-gray-50 cursor-pointer flex items-center gap-3 border-b border-gray-50 last:border-0', `data-id-${acc.id}`]"
            @click="selectAccount(acc)"
          >
            <img :src="acc.image" class="w-8 h-8 rounded-full object-cover" />
            <div>
              <div class="font-bold text-gray-900">{{ acc.name }}</div>
              <div class="text-xs text-gray-500">{{ acc.balance }} {{ acc.currency }}</div>
            </div>
          </div>
        </div>
      </div>

      <!-- To Beneficiary (Display Only) -->
      <div class="mb-6" v-if="beneficiary">
        <label class="block text-sm font-medium text-gray-700 mb-2">To</label>
        <div class="bg-gray-100 rounded-xl p-4 flex items-center gap-3 opacity-80">
          <img :src="beneficiary.image" class="w-8 h-8 rounded-full object-cover grayscale" />
          <div class="font-bold text-gray-900">{{ beneficiary.name }}</div>
        </div>
      </div>

      <!-- Amount Input -->
      <div class="mb-6">
        <label class="block text-sm font-medium text-gray-700 mb-2">Amount</label>
        <div class="relative">
          <span class="absolute left-4 top-1/2 -translate-y-1/2 text-gray-500 font-bold">$</span>
          <input 
            id="input-amount"
            type="number" 
            v-model="amount"
            @input="updateAmount"
            placeholder="0.00"
            class="w-full pl-8 pr-4 py-4 bg-white border border-gray-300 rounded-xl text-2xl font-bold text-gray-900 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all shadow-sm"
          />
        </div>
      </div>

      <!-- Reference Input -->
      <div class="mb-8">
        <label class="block text-sm font-medium text-gray-700 mb-2">Reference</label>
        <input 
          id="input-reference"
          type="text" 
          v-model="reference"
          @input="updateReference"
          placeholder="What's this for?"
          class="w-full px-4 py-3 bg-white border border-gray-300 rounded-xl text-gray-900 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all shadow-sm"
        />
      </div>

      <!-- Continue Button -->
      <button 
        id="cta-continue-transfer"
        @click="continueToReview"
        :disabled="!isValid"
        :class="['w-full py-4 rounded-xl font-bold shadow-lg transition-all', isValid ? 'bg-blue-600 hover:bg-blue-700 text-white shadow-blue-200 active:scale-95' : 'bg-gray-300 text-gray-500 cursor-not-allowed']"
      >
        Continue
      </button>

    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'TRANSFER_FORM',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const showAccountDropdown = ref(false)
    const amount = ref('')
    const reference = ref('')

    const accounts = computed(() => dataStore.accounts)
    
    // Initialize selected account if present in store, else default to first
    const selectedAccount = computed(() => {
      if (signatureStore.from_account_id) {
        return accounts.value.find(a => a.id === signatureStore.from_account_id)
      }
      return null
    })

    const beneficiary = computed(() => {
      return dataStore.beneficiaries.find(b => b.id === signatureStore.payments_selected_beneficiary_id)
    })

    const isValid = computed(() => {
      return signatureStore.from_account_id && 
             amount.value && 
             parseFloat(amount.value) > 0
    })

    const selectAccount = (acc) => {
      signatureStore.from_account_id = acc.id
      showAccountDropdown.value = false
    }

    const updateAmount = (e) => {
      // Direct update to store as per action ACT_TRANSFER_TYPE_AMOUNT effect
      // Actually FSM says set "placeholder" value, but in reality we set actual text
      // We'll update the store variable
      signatureStore.transfer_amount = e.target.value
    }

    const updateReference = (e) => {
      signatureStore.transfer_reference = e.target.value
    }

    const goBack = () => {
      signatureStore.setCurrentPageId('PAYMENTS_LIST')
      router.push({ name: 'PAYMENTS_LIST' })
    }

    const continueToReview = () => {
      if (!isValid.value) return
      signatureStore.setCurrentPageId('TRANSFER_REVIEW')
      router.push({ name: 'TRANSFER_REVIEW' })
    }

    return {
      showAccountDropdown,
      accounts,
      selectedAccount,
      beneficiary,
      amount,
      reference,
      isValid,
      selectAccount,
      updateAmount,
      updateReference,
      goBack,
      continueToReview
    }
  }
}
</script>