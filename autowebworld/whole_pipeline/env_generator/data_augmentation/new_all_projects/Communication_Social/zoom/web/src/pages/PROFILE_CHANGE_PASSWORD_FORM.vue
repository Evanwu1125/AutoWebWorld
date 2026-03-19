<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center p-4">
    <div class="bg-white rounded-xl shadow-lg max-w-md w-full p-8">
      <h1 class="text-2xl font-bold text-gray-900 mb-6">Change Password</h1>
      
      <div class="space-y-5 mb-8">
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-1">Old Password</label>
          <input 
            id="change-old-password-input"
            v-model="oldPass"
            @input="updateOld"
            type="password" 
            class="w-full border border-gray-300 rounded-md px-4 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
          />
        </div>

        <div>
          <label class="block text-sm font-medium text-gray-700 mb-1">New Password</label>
          <input 
            id="change-new-password-input"
            v-model="newPass"
            @input="updateNew"
            type="password" 
            class="w-full border border-gray-300 rounded-md px-4 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
          />
        </div>

        <div>
          <label class="block text-sm font-medium text-gray-700 mb-1">Confirm Password</label>
          <input 
            id="change-confirm-password-input"
            v-model="confirmPass"
            @input="updateConfirm"
            type="password" 
            class="w-full border border-gray-300 rounded-md px-4 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
          />
        </div>
      </div>

      <div class="flex gap-4">
        <button 
          id="change-password-back-profile" 
          @click="goBack"
          class="flex-1 py-2 border border-gray-300 rounded-md text-gray-700 font-medium hover:bg-gray-50"
        >
          Cancel
        </button>
        <button 
          id="change-password-save-button" 
          @click="savePassword"
          class="flex-1 py-2 bg-blue-600 text-white rounded-md font-medium hover:bg-blue-700 disabled:opacity-50"
          :disabled="!isValid"
        >
          Save
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'PROFILE_CHANGE_PASSWORD_FORM',
  setup() {
    const router = useRouter();
    const store = useSignatureStore();

    const oldPass = computed({
      get: () => store.old_password,
      set: (val) => store.old_password = val
    });

    const newPass = computed({
      get: () => store.new_password,
      set: (val) => store.new_password = val
    });

    const confirmPass = computed({
      get: () => store.confirm_password,
      set: (val) => store.confirm_password = val
    });

    const isValid = computed(() => {
      return store.old_password?.length > 0 && 
             store.new_password?.length > 0 && 
             store.confirm_password?.length > 0;
    });

    const updateOld = (e) => store.handleAction('ACT_CHANGE_PASSWORD_TYPE_OLD', { input_text: e.target.value });
    const updateNew = (e) => store.handleAction('ACT_CHANGE_PASSWORD_TYPE_NEW', { input_text: e.target.value });
    const updateConfirm = (e) => store.handleAction('ACT_CHANGE_PASSWORD_TYPE_CONFIRM', { input_text: e.target.value });

    const savePassword = async () => {
      if (store.handleAction('ACT_CHANGE_PASSWORD_SAVE')) {
        await router.push({ name: 'CHANGE_PASSWORD_SUCCESS' });
      }
    };

    const goBack = async () => {
      if (store.handleAction('ACT_CHANGE_PASSWORD_BACK_PROFILE')) {
        await router.push({ name: 'PROFILE_OVERVIEW' });
      }
    };

    return {
      store,
      oldPass,
      newPass,
      confirmPass,
      isValid,
      updateOld,
      updateNew,
      updateConfirm,
      savePassword,
      goBack
    };
  }
}
</script>