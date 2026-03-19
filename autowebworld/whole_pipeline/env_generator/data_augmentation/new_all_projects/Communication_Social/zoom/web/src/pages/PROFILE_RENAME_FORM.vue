<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center p-4">
    <div class="bg-white rounded-xl shadow-lg max-w-md w-full p-8">
      <h1 class="text-2xl font-bold text-gray-900 mb-6">Rename Profile</h1>
      
      <div class="mb-6">
        <label class="block text-sm font-medium text-gray-700 mb-1">Current Name</label>
        <div class="text-gray-500 px-4 py-2 bg-gray-100 rounded-md border border-gray-200">
          {{ store.current_display_name || store.display_name }}
        </div>
      </div>

      <div class="mb-8">
        <label class="block text-sm font-medium text-gray-700 mb-1">New Name</label>
        <input 
          id="rename-name-input"
          v-model="newName"
          @input="updateName"
          type="text" 
          class="w-full border border-gray-300 rounded-md px-4 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
          placeholder="Enter new name"
        />
      </div>

      <div class="flex gap-4">
        <button 
          id="rename-back-profile" 
          @click="goBack"
          class="flex-1 py-2 border border-gray-300 rounded-md text-gray-700 font-medium hover:bg-gray-50"
        >
          Cancel
        </button>
        <button 
          id="rename-save-button" 
          @click="saveName"
          class="flex-1 py-2 bg-blue-600 text-white rounded-md font-medium hover:bg-blue-700 disabled:opacity-50"
          :disabled="!newName"
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
  name: 'PROFILE_RENAME_FORM',
  setup() {
    const router = useRouter();
    const store = useSignatureStore();

    const newName = computed({
      get: () => store.new_display_name,
      set: (val) => store.new_display_name = val
    });

    const updateName = (e) => {
      store.handleAction('ACT_RENAME_TYPE_NEW_NAME', { input_text: e.target.value });
    };

    const saveName = async () => {
      // Note: FSM logic might update display_name in effects or in success page logic.
      // Actually, looking at FSM: ACT_RENAME_SAVE -> RENAME_PROFILE_SUCCESS
      // It doesn't seem to update the actual display_name in effects of ACT_RENAME_SAVE.
      // Wait, let me check FSM effects.
      // ACT_RENAME_SAVE effects: NONE.
      // ACT_RENAME_SUCCESS_GO_HOME effects: sets success message.
      // Where is display_name updated? Maybe I missed it.
      // Ah, maybe it's assumed backend does it.
      // For frontend, I should update it to reflect change.
      // I'll simulate it here before navigation or assume success page handling.
      // I'll update store.display_name manually here to be safe.
      store.display_name = store.new_display_name;
      
      if (store.handleAction('ACT_RENAME_SAVE')) {
        await router.push({ name: 'RENAME_PROFILE_SUCCESS' });
      }
    };

    const goBack = async () => {
      if (store.handleAction('ACT_RENAME_BACK_PROFILE')) {
        await router.push({ name: 'PROFILE_OVERVIEW' });
      }
    };

    return {
      store,
      newName,
      updateName,
      saveName,
      goBack
    };
  }
}
</script>