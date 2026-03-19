<template>
  <div class="min-h-screen bg-slate-50 py-12">
    <div class="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
       <!-- Steps Indicator -->
      <nav aria-label="Progress" class="mb-12">
        <ol role="list" class="space-y-4 md:flex md:space-y-0 md:space-x-8">
          <li class="md:flex-1">
            <div class="group pl-4 py-2 border-l-4 border-blue-600 flex flex-col border-t-0 md:pl-0 md:pt-4 md:pb-0 md:border-l-0 md:border-t-4 opacity-50">
              <span class="text-xs text-blue-600 font-semibold tracking-wide uppercase">Step 1</span>
              <span class="text-sm font-medium">Campaign Info</span>
            </div>
          </li>
          <li class="md:flex-1">
            <div class="group pl-4 py-2 border-l-4 border-blue-600 flex flex-col border-t-0 md:pl-0 md:pt-4 md:pb-0 md:border-l-0 md:border-t-4">
              <span class="text-xs text-blue-600 font-semibold tracking-wide uppercase">Step 2</span>
              <span class="text-sm font-medium">Recipients</span>
            </div>
          </li>
          <li class="md:flex-1">
            <div class="group pl-4 py-2 border-l-4 border-slate-200 flex flex-col border-t-0 md:pl-0 md:pt-4 md:pb-0 md:border-l-0 md:border-t-4">
              <span class="text-xs text-slate-500 font-semibold tracking-wide uppercase">Step 3</span>
              <span class="text-sm font-medium">Content</span>
            </div>
          </li>
        </ol>
      </nav>

      <div class="bg-white shadow rounded-lg overflow-hidden">
        <div class="px-4 py-5 sm:p-6 space-y-8">
          <h2 class="text-xl font-bold text-slate-900">Who are you sending to?</h2>
          
          <!-- Lists Dropdown -->
          <div class="relative">
            <label class="block text-sm font-medium text-slate-700 mb-2">Select a List</label>
            <div id="dropdown-lists">
              <button
                @click="toggleListsDropdown"
                class="w-full bg-white border border-slate-300 rounded-lg py-3 px-4 flex items-center justify-between shadow-sm hover:border-blue-500 focus:outline-none transition-colors"
              >
                <span class="block truncate font-medium text-slate-700">
                  {{ selectedListLabel || 'Choose a list...' }}
                </span>
                <span class="pointer-events-none flex items-center">
                  <svg class="h-5 w-5 text-slate-400" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 3a1 1 0 01.707.293l3 3a1 1 0 01-1.414 1.414L10 5.414 7.707 7.707a1 1 0 01-1.414-1.414l3-3A1 1 0 0110 3zm-3.707 9.293a1 1 0 011.414 0L10 14.586l2.293-2.293a1 1 0 011.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clip-rule="evenodd" /></svg>
                </span>
              </button>

              <div
                v-if="listsDropdownOpen"
                class="absolute z-10 mt-1 w-full bg-white shadow-xl max-h-60 rounded-lg py-1 ring-1 ring-black ring-opacity-5 overflow-auto"
              >
                <div
                  v-for="list in lists"
                  :key="list.id"
                  :class="`option-list-${list.id.replace('list_', '')}`"
                  @click="selectList(list.id)"
                  class="cursor-pointer select-none relative py-2 pl-4 pr-9 hover:bg-blue-50 hover:text-blue-900 transition-colors"
                >
                   <!-- ^ FSM expects .option-list-1 so `option-${list.id}` works if id is list_1 -->
                  <div class="flex items-center justify-between">
                    <span class="font-normal block truncate">{{ list.name }}</span>
                    <span class="text-xs text-slate-500">{{ list.size }} members</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Segments Dropdown -->
          <div class="relative">
             <label class="block text-sm font-medium text-slate-700 mb-2">Or Select a Segment</label>
            <button 
              id="dropdown-segments"
              @click="toggleSegmentsDropdown"
              class="w-full bg-white border border-slate-300 rounded-lg py-3 px-4 flex items-center justify-between shadow-sm hover:border-blue-500 focus:outline-none transition-colors"
            >
              <span class="block truncate font-medium text-slate-700">
                {{ selectedSegmentLabel || 'Choose a segment...' }}
              </span>
               <span class="pointer-events-none flex items-center">
                <svg class="h-5 w-5 text-slate-400" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 3a1 1 0 01.707.293l3 3a1 1 0 01-1.414 1.414L10 5.414 7.707 7.707a1 1 0 01-1.414-1.414l3-3A1 1 0 0110 3zm-3.707 9.293a1 1 0 011.414 0L10 14.586l2.293-2.293a1 1 0 011.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clip-rule="evenodd" /></svg>
              </span>
            </button>

             <div v-if="segmentsDropdownOpen" class="absolute z-10 mt-1 w-full bg-white shadow-xl max-h-60 rounded-lg py-1 ring-1 ring-black ring-opacity-5 overflow-auto">
              <div 
                v-for="segment in segments"
                :key="segment.id"
                :class="`option-${segment.id}`"
                @click="selectSegment(segment.id)"
                class="cursor-pointer select-none relative py-2 pl-4 pr-9 hover:bg-blue-50 hover:text-blue-900 transition-colors"
              >
                <div class="flex items-center justify-between">
                  <span class="font-normal block truncate">{{ segment.name }}</span>
                  <span class="text-xs text-slate-500">{{ segment.size }} members</span>
                </div>
              </div>
            </div>
          </div>

        </div>
        <div class="px-4 py-4 bg-slate-50 border-t border-slate-200 sm:px-6 flex justify-between">
          <button 
            id="back-basics"
            @click="goBack"
            class="inline-flex justify-center py-2 px-4 border border-slate-300 shadow-sm text-sm font-medium rounded-md text-slate-700 bg-white hover:bg-slate-50 focus:outline-none"
          >
            Back
          </button>
          <button 
            id="btn-recipients-continue"
            @click="goContinue"
            :disabled="!isValid"
            class="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Continue to Content
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
  name: 'CREATE_CAMPAIGN_RECIPIENTS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const listsDropdownOpen = ref(false)
    const segmentsDropdownOpen = ref(false)

    const lists = computed(() => dataStore.lists.filter(l => l.type === 'list'))
    const segments = computed(() => dataStore.segments)

    const selectedListLabel = computed(() => {
      if (!store.selected_list_id) return ''
      const list = lists.value.find(l => l.id === store.selected_list_id)
      return list ? list.name : ''
    })

    const selectedSegmentLabel = computed(() => {
      if (!store.selected_segment_id) return ''
      const seg = segments.value.find(s => s.id === store.selected_segment_id)
      return seg ? seg.name : ''
    })

    function toggleListsDropdown() {
      listsDropdownOpen.value = !listsDropdownOpen.value
      if (listsDropdownOpen.value) segmentsDropdownOpen.value = false
    }

    function toggleSegmentsDropdown() {
      segmentsDropdownOpen.value = !segmentsDropdownOpen.value
      if (segmentsDropdownOpen.value) listsDropdownOpen.value = false
    }

    function selectList(id) {
      store.selected_list_id = id
      listsDropdownOpen.value = false
    }

    function selectSegment(id) {
      store.selected_segment_id = id
      segmentsDropdownOpen.value = false
    }

    const isValid = computed(() => {
      // Precondition: selected_list_id length_gt 0
      // FSM only checks list, but logically one or the other should be fine. 
      // STRICT FSM check: "path": "$.selected_list_id", "cond": "length_gt", "value": 0
      // So I MUST enforce list selection? 
      // Wait, let me check FSM again.
      // Line 1083: precondition check for selected_list_id.
      // It does NOT check for segment id.
      // So FSM requires a list to be selected. Segment is optional or alternative but the precondition specifically checks list.
      // I will follow FSM strictness: button disabled if list not selected.
      return store.selected_list_id && store.selected_list_id.length > 0
    })

    async function goBack() {
      store.setCurrentPageId('CREATE_CAMPAIGN_BASICS')
      await router.push({ name: 'CREATE_CAMPAIGN_BASICS' })
    }

    async function goContinue() {
      if (!isValid.value) return
      store.setCurrentPageId('CREATE_CAMPAIGN_CONTENT')
      await router.push({ name: 'CREATE_CAMPAIGN_CONTENT' })
    }

    return {
      lists,
      segments,
      listsDropdownOpen,
      segmentsDropdownOpen,
      selectedListLabel,
      selectedSegmentLabel,
      toggleListsDropdown,
      toggleSegmentsDropdown,
      selectList,
      selectSegment,
      isValid,
      goBack,
      goContinue
    }
  }
}
</script>