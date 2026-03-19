<template>
  <div class="min-h-screen bg-[#FAFBFC] flex items-center justify-center py-12 px-4">
    <div class="max-w-md w-full bg-white p-8 rounded-lg shadow-md border border-gray-200">
      <h2 class="text-2xl font-bold text-[#172B4D] mb-6 text-center">Review Invitation</h2>
      
      <div class="space-y-4 mb-8 bg-gray-50 p-6 rounded-md border border-gray-100">
         <div class="flex justify-between border-b border-gray-200 pb-2">
           <span class="text-gray-500 font-medium">Email</span>
           <span class="text-[#172B4D] font-bold">{{ signatureStore.invite_email }}</span>
         </div>
         <div class="flex justify-between">
           <span class="text-gray-500 font-medium">Role</span>
           <span class="capitalize bg-blue-100 text-blue-800 px-2 py-0.5 rounded text-sm font-bold">{{ signatureStore.invite_role }}</span>
         </div>
      </div>

      <div class="flex space-x-4">
        <button 
          id="invite-member-review-back" 
          @click="goBack"
          class="flex-1 py-2 px-4 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none"
        >
          Back
        </button>
        <button 
          id="invite-member-confirm" 
          @click="confirm"
          class="flex-1 py-2 px-4 border border-transparent rounded-md text-sm font-medium text-white bg-[#0052CC] hover:bg-blue-700 focus:outline-none shadow-sm"
        >
          Send Invite
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'INVITE_MEMBER_REVIEW',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const goBack = async () => {
      signatureStore.currentPageId = 'INVITE_MEMBER_FORM'
      await router.push({ name: 'INVITE_MEMBER_FORM' })
    }

    const confirm = async () => {
      const newMember = {
        id: `user_${Date.now()}`,
        name: signatureStore.invite_email.split('@')[0], // Mock name
        email: signatureStore.invite_email,
        role: signatureStore.invite_role,
        active: 0,
        image: '/images/photo1765608934.jpg'
      }
      
      dataStore.members.push(newMember)
      signatureStore.success_message = `Invitation sent to ${newMember.email}!`
      
      signatureStore.currentPageId = 'INVITE_USER_SUCCESS'
      await router.push({ name: 'INVITE_USER_SUCCESS' })
    }

    return {
      signatureStore,
      goBack,
      confirm
    }
  }
}
</script>