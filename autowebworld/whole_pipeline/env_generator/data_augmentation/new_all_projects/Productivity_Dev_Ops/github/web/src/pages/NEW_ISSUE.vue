<template>
  <div class="min-h-screen bg-[#0d1117] text-white font-sans flex flex-col items-center py-12">
     <div class="w-full max-w-4xl px-6">
        <h1 class="text-2xl font-normal mb-6">New Issue</h1>
        
        <div class="flex flex-col md:flex-row gap-6">
            <!-- Avatar -->
            <div class="hidden md:block">
                <img src="/images/photo1764838110.jpg" class="w-12 h-12 rounded-full border border-gray-700" />
            </div>

            <!-- Form -->
            <div class="flex-grow bg-[#161b22] border border-gray-700 rounded-md">
                <div class="p-4 space-y-4">
                    <input 
                      id="issue_title"
                      type="text" 
                      v-model="issueTitle"
                      placeholder="Title"
                      class="w-full px-3 py-2 bg-[#0d1117] border border-gray-600 rounded-md focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none font-semibold text-lg"
                    />

                    <textarea 
                      id="issue_body"
                      v-model="issueBody"
                      placeholder="Leave a comment"
                      class="w-full bg-[#0d1117] border border-gray-600 rounded-md p-3 min-h-[200px] focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none"
                    ></textarea>
                </div>
                
                <div class="p-4 bg-[#161b22] border-t border-gray-700 flex justify-between items-center rounded-b-md">
                    <button id="cancel-new-issue" @click="goBack" class="text-sm text-gray-400 hover:text-blue-400 hover:underline">
                        Cancel
                    </button>
                    <button 
                      id="submit-new-issue" 
                      @click="submitIssue"
                      :disabled="!issueTitle"
                      :class="['px-4 py-2 text-white font-medium rounded-md', issueTitle ? 'bg-[#238636] hover:bg-[#2ea44f]' : 'bg-gray-700 cursor-not-allowed opacity-50']"
                    >
                        Submit new issue
                    </button>
                </div>
            </div>

            <!-- Sidebar Controls -->
            <div class="w-full md:w-64 space-y-4">
                <div class="relative group">
                    <button id="labels-dropdown" class="w-full text-left text-sm text-gray-400 hover:text-blue-400 flex justify-between items-center" @click="labelsOpen = !labelsOpen">
                        <span class="font-semibold text-gray-300">Labels</span>
                        <span>⚙</span>
                    </button>
                    <div class="text-xs text-gray-500 mt-1">
                        {{ selectedLabels.length ? selectedLabels.join(', ') : 'None yet' }}
                    </div>

                    <!-- Dropdown -->
                    <div v-if="labelsOpen" class="absolute top-8 left-0 w-full bg-[#161b22] border border-gray-700 rounded-md shadow-xl z-10">
                        <div class="px-3 py-2 text-xs font-semibold bg-[#21262d] border-b border-gray-700 rounded-t-md">Apply labels</div>
                        <div id="label-bug" @click="toggleLabel('bug')" class="px-4 py-2 text-sm hover:bg-[#0d1117] cursor-pointer flex items-center gap-2">
                            <span class="w-3 h-3 rounded-full bg-red-500"></span> bug
                        </div>
                        <div id="label-enhancement" @click="toggleLabel('enhancement')" class="px-4 py-2 text-sm hover:bg-[#0d1117] cursor-pointer flex items-center gap-2">
                            <span class="w-3 h-3 rounded-full bg-blue-500"></span> enhancement
                        </div>
                         <div id="label-question" @click="toggleLabel('question')" class="px-4 py-2 text-sm hover:bg-[#0d1117] cursor-pointer flex items-center gap-2">
                            <span class="w-3 h-3 rounded-full bg-purple-500"></span> question
                        </div>
                    </div>
                </div>

                <hr class="border-gray-700" />

                <div class="relative group" id="assignees-menu" @mouseenter="assigneesOpen = true" @mouseleave="assigneesOpen = false">
                     <button class="w-full text-left text-sm text-gray-400 hover:text-blue-400 flex justify-between items-center">
                        <span class="font-semibold text-gray-300">Assignees</span>
                        <span>⚙</span>
                    </button>
                    <div class="text-xs text-gray-500 mt-1">
                         {{ assignee || 'No one' }}
                    </div>

                     <!-- Hover Menu -->
                    <div v-if="assigneesOpen" class="absolute top-8 left-0 w-full bg-[#161b22] border border-gray-700 rounded-md shadow-xl z-10">
                        <div class="px-3 py-2 text-xs font-semibold bg-[#21262d] border-b border-gray-700 rounded-t-md">Assign to</div>
                        <div id="assignee-octocat" @click="setAssignee('octocat')" class="px-4 py-2 text-sm hover:bg-[#0d1117] cursor-pointer flex items-center gap-2">
                            <img src="/images/photo1764838110.jpg" class="w-5 h-5 rounded-full" /> octocat
                        </div>
                        <div id="assignee-hubot" @click="setAssignee('hubot')" class="px-4 py-2 text-sm hover:bg-[#0d1117] cursor-pointer flex items-center gap-2">
                            <img src="/images/User.jpg" class="w-5 h-5 rounded-full" /> hubot
                        </div>
                    </div>
                </div>
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
  name: 'NEW_ISSUE',
  setup() {
    const router = useRouter();
    const store = useSignatureStore();
    const fsmRuntime = new FSMRuntime(fsmData, { store, router });

    const issueTitle = ref('');
    const issueBody = ref('');
    const labelsOpen = ref(false);
    const assigneesOpen = ref(false);
    const selectedLabels = ref([]);
    const assignee = ref(null);

    // Sync to store
    watch(issueTitle, val => store.signature.new_issue_title = val);
    watch(issueBody, val => store.signature.new_issue_body = val);
    
    const toggleLabel = (label) => {
        if (selectedLabels.value.includes(label)) {
            selectedLabels.value = selectedLabels.value.filter(l => l !== label);
        } else {
            selectedLabels.value.push(label);
        }
        // Store expects array of objects {name: 'bug'} per FSM effect, but checking the effect it sets an array of objects
        // ACT_NEW_ISSUE_SELECT_LABEL sets it to [{name: 'bug'}] hardcoded for the action effect
        // We should map our selection to that structure if we want strict compliance, but primarily we just need to trigger the action
        // Here we manually update the signature to reflect current UI state for realism
        store.signature.new_issue_labels = selectedLabels.value.map(l => ({ name: l }));
        labelsOpen.value = false;
    };

    const setAssignee = (user) => {
        assignee.value = user;
        store.signature.new_issue_assignee = user;
        assigneesOpen.value = false;
    };

    const submitIssue = async () => {
        const action = fsmData.pages.find(p => p.id === 'NEW_ISSUE').actions.find(a => a.id === 'ACT_NEW_ISSUE_SUBMIT');
        if (action && fsmRuntime.checkPreconditions(action, store.signature)) {
            store.setCurrentPageId('ISSUE_CREATE_SUCCESS');
            await router.push({ name: 'ISSUE_CREATE_SUCCESS' });
        }
    };

    const goBack = async () => {
        store.setCurrentPageId('ISSUES_LIST');
        await router.push({ name: 'ISSUES_LIST' });
    };

    return {
        issueTitle,
        issueBody,
        labelsOpen,
        assigneesOpen,
        selectedLabels,
        assignee,
        toggleLabel,
        setAssignee,
        submitIssue,
        goBack
    };
  }
}
</script>