<template>
  <div class="h-screen flex flex-col bg-gray-50">
    <!-- Header -->
    <header class="bg-[#6264A7] text-white p-4 shadow-md flex justify-between items-center z-20">
      <div class="font-bold text-lg flex items-center">
        <button id="back-to-teams-from-create" @click="goBack" class="mr-4 hover:bg-[#464775] p-1 rounded">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
          </svg>
        </button>
        Create a team
      </div>
    </header>

    <main class="flex-1 flex flex-col p-8 items-center justify-center">
      <div class="bg-white rounded-lg shadow-lg p-8 w-full max-w-2xl border border-gray-200">
        <div class="flex items-center gap-4 mb-8">
          <div class="bg-purple-100 p-3 rounded-lg">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8 text-[#6264A7]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
            </svg>
          </div>
          <div>
            <h1 class="text-2xl font-bold text-gray-800">Create a team</h1>
            <p class="text-gray-500">Bring everyone together to get work done.</p>
          </div>
        </div>

        <div class="space-y-6">
          <!-- Team Name Input ACT_CREATE_TEAM_TYPE_NAME -->
          <div>
            <label class="block text-sm font-semibold text-gray-700 mb-1">Team name</label>
            <input 
              id="create-team-name-input"
              type="text" 
              v-model="teamName"
              placeholder="e.g. Project Alpha"
              class="w-full rounded-md border-gray-300 shadow-sm focus:border-[#6264A7] focus:ring-[#6264A7] px-4 py-2 border"
            />
          </div>

          <!-- Description Input ACT_CREATE_TEAM_TYPE_DESCRIPTION -->
          <div>
            <label class="block text-sm font-semibold text-gray-700 mb-1">Description</label>
            <textarea 
              id="create-team-description-input"
              v-model="description"
              rows="3"
              placeholder="What is this team about?"
              class="w-full rounded-md border-gray-300 shadow-sm focus:border-[#6264A7] focus:ring-[#6264A7] px-4 py-2 border resize-none"
            ></textarea>
          </div>

          <!-- Privacy Dropdown ACT_CREATE_TEAM_SELECT_PRIVACY -->
          <div>
            <label class="block text-sm font-semibold text-gray-700 mb-1">Privacy</label>
            <div id="create-team-privacy-dropdown" class="relative">
              <button 
                @click="togglePrivacy"
                class="w-full text-left bg-white border border-gray-300 rounded-md px-4 py-2 shadow-sm focus:outline-none focus:ring-1 focus:ring-[#6264A7] flex justify-between items-center"
              >
                <span v-if="privacy">{{ privacy === 'private' ? 'Private - Only team owners can add members' : 'Public - Anyone in your org can join' }}</span>
                <span v-else class="text-gray-400">Select privacy level...</span>
                <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-gray-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" />
                </svg>
              </button>

              <div v-if="privacyOpen" class="absolute z-10 mt-1 w-full bg-white shadow-lg max-h-60 rounded-md py-1 text-base ring-1 ring-black ring-opacity-5 overflow-auto focus:outline-none sm:text-sm">
                <div 
                  id="privacy-private-option"
                  @click="selectPrivacy('private')"
                  class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-gray-100"
                >
                  <div class="flex items-center">
                    <span class="font-medium block truncate">Private</span>
                    <span class="text-gray-500 ml-2 truncate text-xs">Only team owners can add members</span>
                  </div>
                </div>
                <div 
                  id="privacy-public-option"
                  @click="selectPrivacy('public')"
                  class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-gray-100"
                >
                  <div class="flex items-center">
                    <span class="font-medium block truncate">Public</span>
                    <span class="text-gray-500 ml-2 truncate text-xs">Anyone in your org can join</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div class="mt-8 flex justify-end">
          <button 
            id="create-team-submit-button"
            @click="createTeam"
            :disabled="!isValid"
            class="bg-[#6264A7] hover:bg-[#464775] text-white font-semibold py-2 px-6 rounded shadow-sm disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Create
          </button>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { ref, computed, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'CREATE_TEAM',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const teamName = ref('')
    const description = ref('')
    const privacy = ref('')
    const privacyOpen = ref(false)

    const isValid = computed(() => {
      return teamName.value.trim().length > 0 && privacy.value.length > 0
    })

    // Watch for changes and sync to store
    watch(teamName, (val) => {
      store.team_name = val
    })

    watch(description, (val) => {
      store.team_description = val
    })

    const togglePrivacy = () => {
      privacyOpen.value = !privacyOpen.value
    }

    const selectPrivacy = (val) => {
      privacy.value = val;
      privacyOpen.value = false;
      store.privacy_level = val;
    }

    const createTeam = async () => {
      if (!isValid.value) return;

      store.team_name = teamName.value;
      store.team_description = description.value;

      store.currentPageId = 'TEAM_CREATED_SUCCESS';
      await router.push({ name: 'TEAM_CREATED_SUCCESS' });
    }

    const goBack = async () => {
      store.currentPageId = 'TEAMS_LIST';
      await router.push({ name: 'TEAMS_LIST' });
    }

    return {
      teamName,
      description,
      privacy,
      privacyOpen,
      isValid,
      togglePrivacy,
      selectPrivacy,
      createTeam,
      goBack
    }
  }
}
</script>