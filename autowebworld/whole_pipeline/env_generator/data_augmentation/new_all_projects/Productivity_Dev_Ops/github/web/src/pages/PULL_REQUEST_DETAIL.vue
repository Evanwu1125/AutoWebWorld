<template>
  <div class="min-h-screen bg-[#0d1117] text-white font-sans pb-12" v-if="pr">
     <!-- Header -->
    <header class="bg-[#161b22] border-b border-gray-700 py-4 px-6 flex items-center gap-4">
        <div id="pr-back-to-list" class="cursor-pointer text-blue-400 hover:underline" @click="goBack">
             Pull Requests
        </div>
        <span class="text-gray-400">/</span>
        <span class="font-semibold text-gray-300">#{{ pr.id.split('_')[1] }}</span>
    </header>

    <main class="max-w-5xl mx-auto p-6">
        <!-- Title Section -->
        <div class="border-b border-gray-700 pb-6 mb-6">
            <div class="flex items-center justify-between">
                <h1 class="text-3xl font-normal mb-2">{{ pr.title }} <span class="text-gray-500 font-light">#{{ pr.id.split('_')[1] }}</span></h1>
            </div>
            <div class="flex items-center gap-3 text-sm">
                <span v-if="pr.state === 'open'" class="bg-[#238636] text-white px-3 py-1 rounded-full flex items-center gap-1">
                     <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-git-pull-request fill-current"><path d="M1.5 3.25a2.25 2.25 0 1 1 3 2.122v5.256a2.251 2.251 0 1 1-1.5 0V5.372A2.25 2.25 0 0 1 1.5 3.25Zm5.677-.5a3.25 3.25 0 0 1 3.25 3.25v5.256a2.251 2.251 0 1 1-1.5 0V6a1.75 1.75 0 0 0-1.75-1.75H7.227ZM4.5 3.25a.75.75 0 1 0-1.5 0 .75.75 0 0 0 1.5 0ZM3.75 12a.75.75 0 1 0 0 1.5.75.75 0 0 0 0-1.5Zm9.75 0a.75.75 0 1 0 0 1.5.75.75 0 0 0 0-1.5Z"></path></svg>
                     Open
                </span>
                <span v-else-if="pr.state === 'merged'" class="bg-[#8957e5] text-white px-3 py-1 rounded-full flex items-center gap-1">
                     <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-git-merge fill-current"><path d="M5.45 5.13a3.75 3.75 0 0 0-2.7 2.245 3.75 3.75 0 0 0 2.7 4.995V14a.75.75 0 0 0 1.5 0v-1.63a3.75 3.75 0 0 0 2.7-4.995 3.75 3.75 0 0 0-2.7-2.245V2.75a.75.75 0 0 0-1.5 0ZM11.28 2.22a.75.75 0 0 0-1.06 1.06L12.19 5.25H9.25a.75.75 0 0 0 0 1.5h2.94l-1.97 1.97a.75.75 0 1 0 1.06 1.06l3.25-3.25a.75.75 0 0 0 0-1.06Zm-4.58 7.53a2.25 2.25 0 1 1 0-4.5 2.25 2.25 0 0 1 0 4.5Z"></path></svg>
                     Merged
                </span>
                 <span class="text-gray-400">
                    <strong class="text-white">{{ pr.author_id }}</strong> wants to merge {{ pr.head }} into {{ pr.base }}
                </span>
            </div>
        </div>

        <!-- Timeline -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div class="col-span-3 space-y-6">
                <!-- Body -->
                <div class="flex gap-4">
                    <img :src="authorAvatar" class="w-10 h-10 rounded-full border border-gray-700" />
                    <div class="flex-grow border border-gray-700 rounded-md bg-[#161b22]">
                        <div class="bg-[#21262d] px-4 py-2 border-b border-gray-700 rounded-t-md text-sm">
                             <div class="text-gray-300"><strong>{{ pr.author_id }}</strong> commented on {{ pr.created_at }}</div>
                        </div>
                        <div class="p-4 text-gray-300">
                            {{ pr.body }}
                        </div>
                    </div>
                </div>

                <!-- Merge Box -->
                <div class="border border-gray-700 rounded-md bg-[#161b22]">
                     <div class="p-4 flex items-center justify-between bg-[#21262d] rounded-t-md border-b border-gray-700">
                         <div class="flex items-center gap-2">
                            <div class="w-8 h-8 rounded-full bg-green-900/30 flex items-center justify-center">
                                <svg class="w-5 h-5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
                            </div>
                            <div>
                                <h4 class="font-semibold">This branch has no conflicts with the base branch</h4>
                                <p class="text-xs text-gray-400">Merging can be performed automatically.</p>
                            </div>
                         </div>
                         <div class="flex gap-2">
                             <button id="review-approve-button" class="px-3 py-1.5 bg-[#21262d] border border-gray-600 rounded-md hover:bg-gray-700 text-sm">
                                 Approve
                             </button>
                             <button id="merge-pull-request-button" @click="mergePR" v-if="pr.state === 'open'" class="px-3 py-1.5 bg-[#238636] text-white rounded-md hover:bg-[#2ea44f] text-sm font-medium">
                                 Merge pull request
                             </button>
                         </div>
                     </div>
                </div>

                <hr class="border-gray-700" />

                <!-- New Comment Form -->
                <div class="flex gap-4">
                     <img src="/images/Comment.jpg" class="w-10 h-10 rounded-full border border-gray-700" />
                     <div class="flex-grow space-y-2">
                         <textarea 
                           id="new-pr-comment-textarea" 
                           v-model="newComment"
                           class="w-full bg-[#0d1117] border border-gray-600 rounded-md p-3 min-h-[120px] focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none text-gray-300 placeholder-gray-600"
                           placeholder="Leave a comment"
                         ></textarea>
                         <div class="flex justify-end">
                             <button class="px-4 py-2 bg-[#238636] text-white font-medium rounded-md hover:bg-[#2ea44f]">Comment</button>
                         </div>
                     </div>
                </div>
            </div>

             <!-- Sidebar -->
            <div class="space-y-6 text-sm">
                <div class="border-b border-gray-700 pb-4">
                    <h3 class="text-gray-400 font-semibold mb-2">Reviewers</h3>
                    <div class="text-gray-500">No reviews yet</div>
                </div>
                <div class="border-b border-gray-700 pb-4">
                    <h3 class="text-gray-400 font-semibold mb-2">Assignees</h3>
                    <div class="text-gray-500">No one assigned</div>
                </div>
            </div>
        </div>
    </main>
  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';

export default {
  name: 'PULL_REQUEST_DETAIL',
  setup() {
    const router = useRouter();
    const route = useRoute();
    const store = useSignatureStore();
    const dataStore = useDataStore();

    const newComment = ref('');

    const pr = computed(() => {
        const id = route.params.item_id || store.signature.pulls_selected_pr_id;
        return dataStore.pulls.find(p => p.id === id);
    });

    const author = computed(() => {
        if (!pr.value) return null;
        return dataStore.users.find(u => u.id === pr.value.author_id);
    });

    const authorAvatar = computed(() => {
        return author.value?.avatar || '/images/User.jpg';
    });

    const goBack = async () => {
        store.setCurrentPageId('PULL_REQUESTS_LIST');
        await router.push({ name: 'PULL_REQUESTS_LIST' });
    };

    const mergePR = () => {
        if (pr.value) pr.value.state = 'merged';
    };

    onMounted(() => {
        if (!pr.value) {
            // router.push({ name: 'PULL_REQUESTS_LIST' });
        }
    });

    return {
        pr,
        authorAvatar,
        newComment,
        goBack,
        mergePR
    };
  }
}
</script>