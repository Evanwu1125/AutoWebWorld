<template>
  <div class="h-screen bg-slate-50 flex flex-col">
    <header class="bg-white shadow-sm z-20 px-4 py-3 flex items-center justify-between">
        <div class="flex items-center">
            <button id="group-add-members-back" @click="goBackDetails" class="p-2 text-slate-500 hover:text-blue-600 mr-4">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
                </svg>
            </button>
            <div>
                <h1 class="text-xl font-bold text-slate-800">Add Members</h1>
                <p class="text-xs text-slate-500">{{ selectedCount }} selected</p>
            </div>
        </div>
        <button 
            id="group-add-members-next" 
            @click="goNext"
            :disabled="selectedCount === 0"
            class="text-blue-600 font-semibold disabled:opacity-50 disabled:cursor-not-allowed hover:text-blue-700 transition-colors"
        >
            Next
        </button>
    </header>

    <!-- Selected Members Preview -->
    <div v-if="selectedCount > 0" class="bg-white border-b border-slate-100 p-3 flex gap-2 overflow-x-auto">
        <div v-for="id in selectedIds" :key="id" class="flex flex-col items-center flex-shrink-0 w-16">
            <div class="relative">
                <img :src="getContact(id).avatar" class="w-12 h-12 rounded-full object-cover" />
                <div class="absolute -top-1 -right-1 bg-slate-400 text-white rounded-full p-0.5 cursor-pointer hover:bg-slate-600" @click.stop="toggleSelection(id)">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-3 w-3" viewBox="0 0 20 20" fill="currentColor">
                        <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd" />
                    </svg>
                </div>
            </div>
            <span class="text-xs text-slate-600 truncate w-full text-center mt-1">{{ getContact(id).name.split(' ')[0] }}</span>
        </div>
    </div>

    <!-- Contact List -->
    <div id="add-members-list-container" class="flex-1 overflow-y-auto bg-white">
      <div class="max-w-2xl mx-auto divide-y divide-slate-100" id="add-members-list">
        <div 
          v-for="contact in contacts" 
          :key="contact.id"
          class="p-4 hover:bg-slate-50 cursor-pointer transition-colors flex items-center space-x-4 member-row-visible"
          :class="`data-id-${contact.id}`"
          @click="toggleSelection(contact.id)"
        >
            <div class="relative">
                <img :src="contact.avatar" class="w-12 h-12 rounded-full object-cover border border-slate-200" />
                <div v-if="isSelected(contact.id)" class="absolute bottom-0 right-0 bg-blue-600 text-white rounded-full p-1 border-2 border-white">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-3 w-3" viewBox="0 0 20 20" fill="currentColor">
                        <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
                    </svg>
                </div>
            </div>
            
            <div class="flex-1 min-w-0">
                <h3 class="text-base font-semibold text-slate-900 truncate">{{ contact.name }}</h3>
                <p class="text-sm text-slate-500">{{ contact.phone }}</p>
            </div>
            
            <div 
                class="w-6 h-6 rounded-full border-2 flex items-center justify-center transition-colors"
                :class="isSelected(contact.id) ? 'bg-blue-600 border-blue-600' : 'border-slate-300'"
            >
                 <svg v-if="isSelected(contact.id)" xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-white" viewBox="0 0 20 20" fill="currentColor">
                    <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
                 </svg>
            </div>
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
  name: 'GROUP_CREATE_ADD_MEMBERS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const contacts = computed(() => dataStore.contacts)
    const selectedIds = ref([]) // Local reactive state, synced to store on change

    // Sync from store initial state if any
    if (store.selected_member_ids && store.selected_member_ids.length > 0) {
        // Parse if store stores objects as per FSM effect "append(..., {id: ...})"
        // The FSM effect says `append($.selected_member_ids, {"id": "{ITEM_ANY}"})`
        // So store.selected_member_ids is an array of OBJECTS {id: 'xxx'}
        // We need to handle this structure.
        selectedIds.value = store.selected_member_ids.map(item => item.id || item)
    }

    const selectedCount = computed(() => selectedIds.value.length)

    const isSelected = (id) => selectedIds.value.includes(id)

    const getContact = (id) => contacts.value.find(c => c.id === id) || { name: 'Unknown', avatar: '' }

    const toggleSelection = (id) => {
        if (isSelected(id)) {
            selectedIds.value = selectedIds.value.filter(itemId => itemId !== id)
        } else {
            selectedIds.value.push(id)
            // FSM sets viewport anchor ID when scrolling/clicking. 
            // We should clear it if set, to simulate FSM behavior, but mostly handled by action logic.
            store.add_members_viewport_anchor_id = null
        }
        
        // Sync to store in FSM expected format: array of objects {id: 'xxx'}
        // FSM signature says `array<string>|null` in schema, BUT effect uses `append(..., {id: ...})`.
        // This is a discrepancy in FSM. Schema type vs Effect value.
        // Assuming schema `array<string>` is correct type, but effect pushes objects?
        // Let's look at schema: `selected_member_ids: "array<string>|null"`.
        // Let's look at effect: `value_expr: "append($.selected_member_ids, {\"id\": \"{ITEM_ANY}\"})"`.
        // This suggests the store holds objects. I will conform to EFFECT (runtime truth).
        store.selected_member_ids = selectedIds.value.map(id => ({ id }))
    }

    const goBackDetails = async () => {
        store.currentPageId = 'GROUP_CREATE_DETAILS'
        await router.push({ name: 'GROUP_CREATE_DETAILS' })
    }

    const goNext = async () => {
        if (selectedCount.value === 0) return
        store.currentPageId = 'GROUP_CREATE_REVIEW'
        await router.push({ name: 'GROUP_CREATE_REVIEW' })
    }

    return {
        contacts,
        selectedIds,
        selectedCount,
        isSelected,
        getContact,
        toggleSelection,
        goBackDetails,
        goNext
    }
  }
}
</script>