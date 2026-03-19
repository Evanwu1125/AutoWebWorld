<template>
  <div class="min-h-screen bg-slate-50 py-12">
    <div class="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="bg-white shadow rounded-lg overflow-hidden">
        <div class="px-4 py-5 sm:p-6 space-y-6">
          <h2 class="text-xl font-bold text-slate-900">Select SMS List</h2>
          
          <div class="relative" id="sms-dropdown-lists">
            <button 
              @click="toggleDropdown"
              class="w-full bg-white border border-slate-300 rounded-lg py-3 px-4 flex items-center justify-between shadow-sm hover:border-purple-500 focus:outline-none transition-colors"
            >
              <span class="block truncate font-medium text-slate-700">
                {{ selectedLabel || 'Choose a list...' }}
              </span>
              <span class="pointer-events-none flex items-center">
                <svg class="h-5 w-5 text-slate-400" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 3a1 1 0 01.707.293l3 3a1 1 0 01-1.414 1.414L10 5.414 7.707 7.707a1 1 0 01-1.414-1.414l3-3A1 1 0 0110 3zm-3.707 9.293a1 1 0 011.414 0L10 14.586l2.293-2.293a1 1 0 011.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clip-rule="evenodd" /></svg>
              </span>
            </button>

            <div v-if="isOpen" class="absolute z-10 mt-1 w-full bg-white shadow-xl max-h-60 rounded-lg py-1 ring-1 ring-black ring-opacity-5 overflow-auto">
              <div 
                v-for="list in smsLists"
                :key="list.id"
                :class="`option-list-${list.id.replace('list_','')}`"
                @click="selectList(list.id)"
                class="cursor-pointer select-none relative py-2 pl-4 pr-9 hover:bg-purple-50 hover:text-purple-900 transition-colors"
              >
                <div class="flex items-center justify-between">
                  <span class="font-normal block truncate">{{ list.name }}</span>
                  <span class="text-xs text-slate-500">{{ list.size }} members</span>
                </div>
              </div>
            </div>
          </div>

        </div>
        <div class="px-4 py-4 bg-slate-50 border-t border-slate-200 sm:px-6 flex justify-between">
          <button 
            id="back-sms-basics"
            @click="goBack"
            class="inline-flex justify-center py-2 px-4 border border-slate-300 shadow-sm text-sm font-medium rounded-md text-slate-700 bg-white hover:bg-slate-50 focus:outline-none"
          >
            Back
          </button>
          <button 
            id="btn-sms-recipients-continue"
            @click="goContinue"
            :disabled="!isValid"
            class="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-purple-600 hover:bg-purple-700 focus:outline-none disabled:opacity-50"
          >
            Continue
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
import { useDataStore } from '../stores/data'

export default {
  name: 'CREATE_SMS_CAMPAIGN_RECIPIENTS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()
    const isOpen = ref(false)

    // Filter only lists that start with sms_ or generic lists, but for demo just all lists
    const smsLists = computed(() => dataStore.lists)

    const selectedLabel = computed(() => {
      if (!store.sms_selected_list_id) return ''
      const list = smsLists.value.find(l => l.id === store.sms_selected_list_id)
      return list ? list.name : ''
    })

    function toggleDropdown() {
      isOpen.value = !isOpen.value
    }

    function selectList(id) {
      store.sms_selected_list_id = id
      isOpen.value = false
    }

    const isValid = computed(() => {
      return store.sms_selected_list_id && store.sms_selected_list_id.length > 0
    })

    async function goBack() {
      store.setCurrentPageId('CREATE_SMS_CAMPAIGN_BASICS')
      await router.push({ name: 'CREATE_SMS_CAMPAIGN_BASICS' })
    }

    async function goContinue() {
      if (!isValid.value) return
      store.setCurrentPageId('CREATE_SMS_CAMPAIGN_CONTENT')
      await router.push({ name: 'CREATE_SMS_CAMPAIGN_CONTENT' })
    }

    return {
      smsLists,
      isOpen,
      selectedLabel,
      toggleDropdown,
      selectList,
      isValid,
      goBack,
      goContinue
    }
  }
}
</script>