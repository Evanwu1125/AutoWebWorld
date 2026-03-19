<template>
  <div class="min-h-screen bg-slate-50 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
    <div class="max-w-md w-full space-y-8 bg-white p-10 rounded-xl shadow-lg border border-slate-100">
      <div class="text-center">
        <h2 class="text-3xl font-extrabold text-slate-900">Choose Channel</h2>
        <p class="mt-2 text-sm text-slate-600">Select the type of campaign you want to create.</p>
      </div>
      
      <div class="mt-8 space-y-6">
        <div class="relative">
          <button 
            id="channel-dropdown"
            @click="toggleDropdown"
            class="w-full bg-white border border-slate-300 rounded-lg py-3 px-4 flex items-center justify-between shadow-sm hover:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors"
          >
            <span class="block truncate font-medium text-slate-700">
              {{ selectedChannelLabel || 'Select Channel...' }}
            </span>
            <span class="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-4">
              <svg class="h-5 w-5 text-slate-400" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                <path fill-rule="evenodd" d="M10 3a1 1 0 01.707.293l3 3a1 1 0 01-1.414 1.414L10 5.414 7.707 7.707a1 1 0 01-1.414-1.414l3-3A1 1 0 0110 3zm-3.707 9.293a1 1 0 011.414 0L10 14.586l2.293-2.293a1 1 0 011.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clip-rule="evenodd" />
              </svg>
            </span>
          </button>

          <div v-if="isOpen" class="absolute z-10 mt-1 w-full bg-white shadow-xl max-h-60 rounded-lg py-1 text-base ring-1 ring-black ring-opacity-5 overflow-auto focus:outline-none sm:text-sm">
            <div 
              id="channel-email"
              @click="selectChannel('email')"
              class="cursor-pointer select-none relative py-3 pl-4 pr-9 hover:bg-blue-50 hover:text-blue-900 transition-colors"
            >
              <div class="flex items-center">
                <span class="font-normal block truncate">Email Campaign</span>
              </div>
            </div>

            <div 
              id="channel-sms"
              @click="selectChannel('sms')"
              class="cursor-pointer select-none relative py-3 pl-4 pr-9 hover:bg-blue-50 hover:text-blue-900 transition-colors"
            >
              <div class="flex items-center">
                <span class="font-normal block truncate">SMS Campaign</span>
              </div>
            </div>
          </div>
        </div>

        <div class="flex justify-center mt-6">
          <button 
            id="back-campaigns"
            @click="goBack"
            class="text-sm font-medium text-slate-500 hover:text-slate-900 transition-colors"
          >
            Cancel and go back
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'CREATE_CAMPAIGN_CHANNEL',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const isOpen = ref(false)

    const selectedChannelLabel = computed(() => {
      if (store.selected_channel === 'email') return 'Email Campaign'
      if (store.selected_channel === 'sms') return 'SMS Campaign'
      return ''
    })

    function toggleDropdown() {
      isOpen.value = !isOpen.value
    }

    async function selectChannel(channel) {
      store.selected_channel = channel
      if (channel === 'email') {
        store.setCurrentPageId('CREATE_CAMPAIGN_BASICS')
        await router.push({ name: 'CREATE_CAMPAIGN_BASICS' })
      } else {
        store.setCurrentPageId('CREATE_SMS_CAMPAIGN_BASICS')
        await router.push({ name: 'CREATE_SMS_CAMPAIGN_BASICS' })
      }
    }

    async function goBack() {
      store.setCurrentPageId('CAMPAIGNS_LIST')
      await router.push({ name: 'CAMPAIGNS_LIST' })
    }

    return {
      isOpen,
      selectedChannelLabel,
      toggleDropdown,
      selectChannel,
      goBack
    }
  }
}
</script>