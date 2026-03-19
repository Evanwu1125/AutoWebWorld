<template>
  <div class="min-h-screen bg-[#0d1117] text-white font-sans" v-if="repo">
    <!-- Header -->
    <header class="bg-[#161b22] border-b border-gray-700 py-4 px-6">
        <div class="flex items-center gap-2 text-lg">
            <svg height="16" aria-hidden="true" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-repo text-gray-400 fill-current"><path d="M2 2.5A2.5 2.5 0 0 1 4.5 0h8.75a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75h-2.5a.75.75 0 1 1 0-1.5h1.75v-2h-8a1 1 0 0 0-.714 1.7.75.75 0 1 1-1.072 1.05A2.495 2.495 0 0 1 2 11.5Zm10.5-1V9h-8c-.356 0-.694.074-1 .208V2.5a1 1 0 0 1 1-1ZM5 12.25a.25.25 0 0 1 .25-.25h3.5a.25.25 0 0 1 .25.25v3.25a.25.25 0 0 1-.4.2l-1.45-1.087a.25.25 0 0 0-.3 0L5.4 15.7a.25.25 0 0 1-.4-.2Z"></path></svg>
            <span id="repo-back-to-repos" class="text-blue-400 hover:underline cursor-pointer" @click="goBack">Repositories</span>
            <span class="text-gray-400">/</span>
            <span class="font-semibold">{{ repo.name }}</span>
            <span class="ml-2 px-2 py-0.5 text-xs border border-gray-600 rounded-full text-gray-400">{{ repo.private ? 'Private' : 'Public' }}</span>
        </div>
    </header>

    <!-- Tabs -->
    <div class="bg-[#0d1117] border-b border-gray-700 px-6 pt-4">
        <nav class="flex space-x-6">
            <div class="pb-3 border-b-2 border-[#fd8c73] font-semibold flex items-center gap-2 cursor-pointer">
                <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-code"><path d="m11.28 3.22 4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.749.749 0 0 1-1.275-.326.749.749 0 0 1 .215-.734L13.94 8l-3.72-3.72a.749.749 0 0 1 .326-1.275.749.749 0 0 1 .734.215Zm-6.56 0a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042L2.06 8l3.72 3.72a.749.749 0 0 1-.326 1.275.749.749 0 0 1-.734-.215L.47 8.53a.75.75 0 0 1 0-1.06L4.72 3.22a.75.75 0 0 1 1.042-.018Z"></path></svg>
                Code
            </div>
            <div id="repo-tab-issues" @click="goToIssues" class="pb-3 border-b-2 border-transparent hover:border-gray-300 text-gray-300 flex items-center gap-2 cursor-pointer">
                <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-issue-opened"><path d="M8 9.5a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3Z"></path><path d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0ZM1.5 8a6.5 6.5 0 1 0 13 0 6.5 6.5 0 0 0-13 0Z"></path></svg>
                Issues
            </div>
            <div id="repo-tab-pull-requests" @click="goToPulls" class="pb-3 border-b-2 border-transparent hover:border-gray-300 text-gray-300 flex items-center gap-2 cursor-pointer">
                <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-git-pull-request"><path d="M1.5 3.25a2.25 2.25 0 1 1 3 2.122v5.256a2.251 2.251 0 1 1-1.5 0V5.372A2.25 2.25 0 0 1 1.5 3.25Zm5.677-.5a3.25 3.25 0 0 1 3.25 3.25v5.256a2.251 2.251 0 1 1-1.5 0V6a1.75 1.75 0 0 0-1.75-1.75H7.227ZM4.5 3.25a.75.75 0 1 0-1.5 0 .75.75 0 0 0 1.5 0ZM3.75 12a.75.75 0 1 0 0 1.5.75.75 0 0 0 0-1.5Zm9.75 0a.75.75 0 1 0 0 1.5.75.75 0 0 0 0-1.5Z"></path></svg>
                Pull requests
            </div>
        </nav>
    </div>

    <!-- Content -->
    <main class="max-w-6xl mx-auto p-6 grid grid-cols-1 md:grid-cols-4 gap-6">
        <!-- Left Sidebar -->
        <div class="col-span-3 space-y-4">
            <!-- Branch/File Header -->
            <div class="flex items-center justify-between">
                <button id="repo-branches-link" @click="goToBranches" class="px-3 py-1.5 bg-[#21262d] border border-gray-600 rounded-md text-sm font-medium text-gray-300 hover:bg-gray-700 flex items-center gap-2">
                    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-git-branch"><path d="M9.5 3.25a2.25 2.25 0 1 1 3 2.122V6A2.5 2.5 0 0 1 10 8.5H6a1 1 0 0 0-1 1v1.128a2.251 2.251 0 1 1-1.5 0V5.372a2.25 2.25 0 1 1 1.5 0v1.836A2.493 2.493 0 0 1 6 7h4a1 1 0 0 0 1-1v-.628A2.25 2.25 0 0 1 9.5 3.25Zm-6 0a.75.75 0 1 0 1.5 0 .75.75 0 0 0-1.5 0Zm8.25 0a.75.75 0 1 0 0 1.5.75.75 0 0 0 0-1.5ZM4.25 12a.75.75 0 1 0 0 1.5.75.75 0 0 0 0-1.5Z"></path></svg>
                    main
                    <span class="text-xs">▾</span>
                </button>

                <div class="flex items-center gap-2">
                    <button id="new-pull-request-button" @click="goToNewPR" class="px-3 py-1.5 bg-[#21262d] border border-gray-600 rounded-md text-sm font-medium text-gray-300 hover:bg-gray-700">
                        New pull request
                    </button>
                    <button class="px-3 py-1.5 bg-[#238636] rounded-md text-sm font-medium text-white hover:bg-[#2ea44f]">
                        Code ▾
                    </button>
                </div>
            </div>

            <!-- File List (Mock) -->
            <div class="border border-gray-700 rounded-md overflow-hidden bg-[#0d1117]">
                <div class="bg-[#161b22] px-4 py-3 border-b border-gray-700 flex items-center justify-between">
                     <div class="flex items-center gap-2 text-sm text-gray-300">
                         <img :src="ownerAvatar" class="w-5 h-5 rounded-full" />
                         <span class="font-semibold">{{ ownerName }}</span>
                         <span>latest commit</span>
                     </div>
                     <div class="text-sm text-gray-500">3 hours ago</div>
                </div>
                <div class="divide-y divide-gray-700">
                    <div class="px-4 py-2 text-sm hover:bg-[#161b22] flex items-center gap-3">
                        <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-file text-gray-400"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg>
                        <span class="text-gray-300">src</span>
                        <span class="text-gray-500 ml-auto">Initial commit</span>
                    </div>
                    <div class="px-4 py-2 text-sm hover:bg-[#161b22] flex items-center gap-3">
                        <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-file text-gray-400"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg>
                        <span class="text-gray-300">package.json</span>
                        <span class="text-gray-500 ml-auto">Update dependencies</span>
                    </div>
                    <div class="px-4 py-2 text-sm hover:bg-[#161b22] flex items-center gap-3">
                         <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-file text-gray-400"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg>
                        <span class="text-gray-300">README.md</span>
                        <span class="text-gray-500 ml-auto">Update docs</span>
                    </div>
                </div>
            </div>

            <!-- Readme Preview -->
             <div class="border border-gray-700 rounded-md bg-[#0d1117]">
                 <div class="bg-[#161b22] px-4 py-2 border-b border-gray-700 font-semibold text-sm">README.md</div>
                 <div class="p-6 prose prose-invert max-w-none">
                     <h1>{{ repo.name }}</h1>
                     <p>{{ repo.description }}</p>
                     <h2>Getting Started</h2>
                     <pre><code>git clone https://github.com/user/{{repo.name}}.git
cd {{repo.name}}
npm install
npm start</code></pre>
                 </div>
             </div>
        </div>

        <!-- Right Sidebar -->
        <div class="space-y-6">
            <div class="border-b border-gray-700 pb-4">
                <h3 class="font-semibold mb-2">About</h3>
                <p class="text-sm text-gray-400">{{ repo.description }}</p>
                <div class="mt-4 text-sm flex items-center gap-2">
                    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-book text-gray-400"><path d="M0 5.75C0 4.784.784 4 1.75 4h12.5c.966 0 1.75.784 1.75 1.75v8.5A1.75 1.75 0 0 1 14.25 16H1.75A1.75 1.75 0 0 1 0 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v8.5c0 .138.112.25.25.25h12.5a.25.25 0 0 0 .25-.25v-8.5a.25.25 0 0 0-.25-.25ZM3.5 6.25a.75.75 0 0 1 .75.75v6a.75.75 0 0 1-1.5 0v-6a.75.75 0 0 1 .75-.75Zm4.25.75a.75.75 0 0 0-1.5 0v6a.75.75 0 0 0 1.5 0ZM12 7a.75.75 0 0 0-1.5 0v6a.75.75 0 0 0 1.5 0Z"></path></svg>
                    Readme
                </div>
                <div class="mt-2 text-sm flex items-center gap-2">
                     <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-star text-gray-400"><path d="M8 .25a.75.75 0 0 1 .673.418l1.882 3.815 4.21.612a.75.75 0 0 1 .416 1.279l-3.046 2.97.719 4.192a.75.75 0 0 1-1.088.791L8 12.347l-3.766 1.98a.75.75 0 0 1-1.088-.79l.72-4.194L.818 6.374a.75.75 0 0 1 .416-1.28l4.21-.611L7.327.668A.75.75 0 0 1 8 .25Z"></path></svg>
                    <strong>{{ repo.stars }}</strong> stars
                </div>
            </div>

            <div>
                <h3 class="font-semibold mb-2">Languages</h3>
                <div class="flex h-2 rounded-full overflow-hidden mb-2">
                    <div class="bg-yellow-400 w-full"></div>
                </div>
                <div class="flex items-center gap-2 text-xs">
                    <span class="w-2 h-2 rounded-full bg-yellow-400"></span>
                    <span class="font-semibold">{{ repo.language }}</span> 100.0%
                </div>
            </div>
        </div>
    </main>
  </div>
</template>

<script>
import { computed, onMounted } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';

export default {
  name: 'REPOSITORY_DETAIL',
  setup() {
    const router = useRouter();
    const route = useRoute();
    const store = useSignatureStore();
    const dataStore = useDataStore();

    const repo = computed(() => {
      const id = route.params.item_id || store.signature.repos_selected_repo_id;
      return dataStore.repositories.find(r => r.id === id);
    });

    const owner = computed(() => {
      if (!repo.value) return null;
      return dataStore.users.find(u => u.id === repo.value.owner_id);
    });

    const ownerAvatar = computed(() => {
      return owner.value?.avatar || '/images/User.jpg';
    });

    const ownerName = computed(() => {
      return owner.value?.username || 'User';
    });

    const goBack = async () => {
      store.setCurrentPageId('REPOSITORIES_LIST');
      await router.push({ name: 'REPOSITORIES_LIST' });
    };

    const goToIssues = async () => {
      store.setCurrentPageId('ISSUES_LIST');
      await router.push({ name: 'ISSUES_LIST' });
    };

    const goToPulls = async () => {
      store.setCurrentPageId('PULL_REQUESTS_LIST');
      await router.push({ name: 'PULL_REQUESTS_LIST' });
    };

    const goToBranches = async () => {
      store.setCurrentPageId('BRANCHES_LIST');
      await router.push({ name: 'BRANCHES_LIST' });
    };

    const goToNewPR = async () => {
      store.setCurrentPageId('NEW_PULL_REQUEST');
      await router.push({ name: 'NEW_PULL_REQUEST' });
    };

    onMounted(() => {
      // Ensure we have a selected repo
      if (!repo.value) {
         // Fallback or redirect logic
         // router.push({ name: 'REPOSITORIES_LIST' });
      }
    });

    return {
      repo,
      ownerAvatar,
      ownerName,
      goBack,
      goToIssues,
      goToPulls,
      goToBranches,
      goToNewPR
    };
  }
}
</script>