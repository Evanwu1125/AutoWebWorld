<template>
  <div class="min-h-screen bg-[#FAFBFC] flex items-center justify-center py-12 px-4">
    <div class="max-w-md w-full bg-white p-8 rounded-lg shadow-md border border-gray-200">
      <h2 class="text-2xl font-bold text-[#172B4D] mb-6">Invite Workspace Member</h2>
      
      <div class="space-y-6">
        <!-- Email -->
        <div>
          <label for="invite-email-input" class="block text-sm font-medium text-gray-700 mb-1">Email Address <span class="text-red-500">*</span></label>
          <input 
            id="invite-email-input" 
            v-model="email"
            type="email" 
            class="shadow-sm focus:ring-blue-500 focus:border-blue-500 block w-full sm:text-sm border border-gray-300 rounded-md p-2"
            placeholder="colleague@example.com"
          >
        </div>

        <!-- Role -->
        <div class="relative">
          <label class="block text-sm font-medium text-gray-700 mb-1">Role <span class="text-red-500">*</span></label>
          <button 
            id="invite-role-dropdown"
            @click="toggleRole"
            class="w-full bg-white border border-gray-300 rounded-md shadow-sm px-4 py-2 text-left flex justify-between items-center"
          >
            <span class="capitalize">{{ role || 'Select role' }}</span>
            <svg class="h-5 w-5 text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" /></svg>
          </button>
          
          <div v-if="isRoleOpen" class="absolute mt-1 w-full bg-white shadow-lg rounded-md py-1 z-10 border border-gray-100">
             <div id="invite-role-admin" @click="selectRole('admin')" class="px-4 py-2 hover:bg-gray-100 cursor-pointer">Admin</div>
             <div id="invite-role-developer" @click="selectRole('developer')" class="px-4 py-2 hover:bg-gray-100 cursor-pointer">Developer</div>
             <div id="invite-role-viewer" @click="selectRole('viewer')" class="px-4 py-2 hover:bg-gray-100 cursor-pointer">Viewer</div>
          </div>
        </div>

        <!-- Actions -->
        <div class="flex justify-end space-x-4 pt-4 border-t border-gray-200">
           <button 
             id="invite-member-back" 
             @click="goBack"
             class="px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none"
           >
             Cancel
           </button>
           <button 
             id="invite-member-submit" 
             @click="submit"
             :disabled="!isValid"
             class="px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed"
           >
             Review Invite
           </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'INVITE_MEMBER_FORM',
  setup() {
    const router = useRouter()
    const route = useRoute()
    const signatureStore = useSignatureStore()

    const email = ref('')
    const role = ref(null)
    const isRoleOpen = ref(false)

    // Pre-fill if coming from list click? FSM doesn't specify data flow for editing.
    // Assuming new invite for simplicity unless member_id implies editing.
    // If member_id present, could fetch data, but FSM actions are about 'type' and 'select', suggesting new input.
    // I will treat it as a new invite form, possibly 'cloning' or just navigated to.

    const toggleRole = () => isRoleOpen.value = !isRoleOpen.value

    const selectRole = (val) => {
      role.value = val
      signatureStore.invite_role = val
      isRoleOpen.value = false
    }

    const isValid = computed(() => email.value.length > 0 && role.value)

    const submit = async () => {
      signatureStore.invite_email = email.value
      // role set in select
      signatureStore.currentPageId = 'INVITE_MEMBER_REVIEW'
      await router.push({ name: 'INVITE_MEMBER_REVIEW' })
    }

    const goBack = async () => {
      signatureStore.currentPageId = 'WORKSPACE_MEMBERS'
      await router.push({ name: 'WORKSPACE_MEMBERS' })
    }

    return {
      email,
      role,
      isRoleOpen,
      toggleRole,
      selectRole,
      isValid,
      submit,
      goBack
    }
  }
}
</script>