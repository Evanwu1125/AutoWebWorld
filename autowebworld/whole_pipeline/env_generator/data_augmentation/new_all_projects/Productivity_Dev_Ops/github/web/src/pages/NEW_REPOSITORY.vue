<template>
  <div class="min-h-screen bg-[#0d1117] text-white font-sans flex flex-col items-center py-12">
    <div class="w-full max-w-3xl px-6">
      <div class="mb-8">
        <h1 class="text-2xl font-semibold">Create a new repository</h1>
        <p class="text-gray-400 text-sm mt-1">A repository contains all project files, including the revision history.</p>
      </div>

      <div class="bg-[#161b22] border border-gray-700 rounded-md p-6 space-y-6">
        <!-- Repo Name -->
        <div>
            <label class="block font-semibold mb-2">Repository name <span class="text-red-500">*</span></label>
            <input 
              id="repository_name"
              type="text" 
              v-model="repoName"
              class="w-full md:w-1/2 px-3 py-2 bg-[#0d1117] border border-gray-600 rounded-md focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none"
            />
        </div>

        <!-- Description -->
        <div>
            <label class="block font-semibold mb-2">Description <span class="text-gray-400 font-normal">(optional)</span></label>
            <input 
              id="repository_description"
              type="text" 
              v-model="repoDesc"
              class="w-full px-3 py-2 bg-[#0d1117] border border-gray-600 rounded-md focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none"
            />
        </div>

        <hr class="border-gray-700 my-4" />

        <!-- Visibility -->
        <div id="visibility-menu" class="space-y-4">
            <div 
                id="visibility-public"
                class="flex items-start gap-3 cursor-pointer p-2 rounded-md hover:bg-[#1c2128]"
                @click="setVisibility(false)"
            >
                <div class="mt-1">
                    <input type="radio" name="visibility" :checked="!isPrivate" class="text-blue-500 focus:ring-0 bg-gray-800 border-gray-600" />
                </div>
                <div>
                    <div class="font-semibold flex items-center gap-2">
                        <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-repo text-gray-400"><path d="M2 2.5A2.5 2.5 0 0 1 4.5 0h8.75a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75h-2.5a.75.75 0 1 1 0-1.5h1.75v-2h-8a1 1 0 0 0-.714 1.7.75.75 0 1 1-1.072 1.05A2.495 2.495 0 0 1 2 11.5Zm10.5-1V9h-8c-.356 0-.694.074-1 .208V2.5a1 1 0 0 1 1-1ZM5 12.25a.25.25 0 0 1 .25-.25h3.5a.25.25 0 0 1 .25.25v3.25a.25.25 0 0 1-.4.2l-1.45-1.087a.25.25 0 0 0-.3 0L5.4 15.7a.25.25 0 0 1-.4-.2Z"></path></svg>
                        Public
                    </div>
                    <p class="text-sm text-gray-400">Anyone on the internet can see this repository. You choose who can commit.</p>
                </div>
            </div>
            
            <div 
                id="visibility-private"
                class="flex items-start gap-3 cursor-pointer p-2 rounded-md hover:bg-[#1c2128]"
                @click="setVisibility(true)"
            >
                 <div class="mt-1">
                    <input type="radio" name="visibility" :checked="isPrivate" class="text-blue-500 focus:ring-0 bg-gray-800 border-gray-600" />
                </div>
                <div>
                    <div class="font-semibold flex items-center gap-2">
                        <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-lock text-yellow-500"><path d="M4 4a4 4 0 0 1 8 0v2h.25c.966 0 1.75.784 1.75 1.75v5.5A1.75 1.75 0 0 1 12.25 15h-8.5A1.75 1.75 0 0 1 2 13.25v-5.5C2 6.784 2.784 6 3.75 6H4V4Zm4-2.5a2.5 2.5 0 0 0-2.5 2.5v2h5v-2A2.5 2.5 0 0 0 8 1.5ZM3.5 7.5v5.5a.25.25 0 0 0 .25.25h8.5a.25.25 0 0 0 .25-.25v-5.5a.25.25 0 0 0-.25-.25h-8.5a.25.25 0 0 0-.25.25Z"></path></svg>
                        Private
                    </div>
                    <p class="text-sm text-gray-400">You choose who can see and commit to this repository.</p>
                </div>
            </div>
        </div>

        <hr class="border-gray-700 my-4" />

        <!-- Readme Template -->
        <div>
            <label class="block font-semibold mb-2">Add a README file</label>
            <div class="relative inline-block">
                <button 
                    id="readme-template-dropdown" 
                    @click="templateDropdownOpen = !templateDropdownOpen"
                    class="px-3 py-2 bg-[#21262d] border border-gray-600 rounded-md text-sm font-medium text-gray-300 hover:bg-gray-700 flex items-center gap-2"
                >
                    Template: {{ selectedTemplate }} ▾
                </button>
                <div v-if="templateDropdownOpen" class="absolute left-0 z-10 mt-2 w-48 bg-[#161b22] border border-gray-700 rounded-md shadow-lg">
                    <div id="readme-template-default" @click="setTemplate('Default')" class="block px-4 py-2 text-sm text-gray-300 hover:bg-blue-600 hover:text-white cursor-pointer">Default</div>
                    <div id="readme-template-minimal" @click="setTemplate('Minimal')" class="block px-4 py-2 text-sm text-gray-300 hover:bg-blue-600 hover:text-white cursor-pointer">Minimal</div>
                    <div id="readme-template-none" @click="setTemplate('None')" class="block px-4 py-2 text-sm text-gray-300 hover:bg-blue-600 hover:text-white cursor-pointer">None</div>
                </div>
            </div>
        </div>

        <hr class="border-gray-700 my-4" />
        
        <div class="flex items-center gap-4 pt-4">
            <button 
                id="create-repository-submit"
                @click="createRepo"
                :disabled="!repoName"
                :class="['px-4 py-2 text-white rounded-md font-medium', repoName ? 'bg-[#238636] hover:bg-[#2ea44f]' : 'bg-gray-700 cursor-not-allowed opacity-50']"
            >
                Create repository
            </button>
            <button 
                id="cancel-new-repo" 
                @click="goBack"
                class="text-blue-400 hover:underline text-sm"
            >
                Cancel
            </button>
        </div>

      </div>
    </div>
  </div>
</template>

<script>
import { ref, watch } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { FSMRuntime } from '../fsm/FSMRuntime';
import fsmData from '../../fsm.json';

export default {
  name: 'NEW_REPOSITORY',
  setup() {
    const router = useRouter();
    const store = useSignatureStore();
    const fsmRuntime = new FSMRuntime(fsmData, { store, router });

    const repoName = ref('');
    const repoDesc = ref('');
    const isPrivate = ref(false);
    const selectedTemplate = ref('Default');
    const templateDropdownOpen = ref(false);

    // Sync refs to store on change
    watch(repoName, (val) => store.signature.new_repo_name = val);
    watch(repoDesc, (val) => store.signature.new_repo_description = val);

    const setVisibility = (val) => {
        isPrivate.value = val;
        store.signature.new_repo_private = val;
    };

    const setTemplate = (val) => {
        selectedTemplate.value = val;
        store.signature.new_repo_readme_template = val;
        templateDropdownOpen.value = false;
    };

    const createRepo = async () => {
        // Check FSM logic
        const action = fsmData.pages.find(p => p.id === 'NEW_REPOSITORY').actions.find(a => a.id === 'ACT_NEW_REPO_SUBMIT');
        if (action && fsmRuntime.checkPreconditions(action, store.signature)) {
            store.setCurrentPageId('REPO_CREATE_SUCCESS');
            await router.push({ name: 'REPO_CREATE_SUCCESS' });
        }
    };

    const goBack = async () => {
        store.setCurrentPageId('REPOSITORIES_LIST');
        await router.push({ name: 'REPOSITORIES_LIST' });
    };

    return {
        repoName,
        repoDesc,
        isPrivate,
        selectedTemplate,
        templateDropdownOpen,
        setVisibility,
        setTemplate,
        createRepo,
        goBack
    };
  }
}
</script>