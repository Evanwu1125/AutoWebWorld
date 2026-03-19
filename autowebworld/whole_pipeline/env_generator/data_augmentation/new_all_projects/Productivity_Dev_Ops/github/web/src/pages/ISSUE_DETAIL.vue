<template>
  <div class="min-h-screen bg-[#0d1117] text-white font-sans pb-12" v-if="issue">
     <!-- Header -->
    <header class="bg-[#161b22] border-b border-gray-700 py-4 px-6 flex items-center gap-4">
        <div id="issue-back-to-list" class="cursor-pointer text-blue-400 hover:underline" @click="goBack">
             Issues
        </div>
        <span class="text-gray-400">/</span>
        <span class="font-semibold text-gray-300">#{{ issue.id.split('_')[1] }}</span>
    </header>

    <main class="max-w-5xl mx-auto p-6">
        <!-- Title Section -->
        <div class="border-b border-gray-700 pb-6 mb-6">
            <div class="flex items-center justify-between">
                <h1 class="text-3xl font-normal mb-2">{{ issue.title }} <span class="text-gray-500 font-light">#{{ issue.id.split('_')[1] }}</span></h1>
                <div class="flex gap-2">
                    <button id="issue-close-button" v-if="issue.state === 'open'" @click="closeIssue" class="px-3 py-1.5 text-sm bg-gray-800 text-red-400 border border-gray-600 rounded-md hover:bg-gray-700">
                        Close issue
                    </button>
                    <button id="issue-reopen-button" v-else @click="reopenIssue" class="px-3 py-1.5 text-sm bg-gray-800 text-green-400 border border-gray-600 rounded-md hover:bg-gray-700">
                        Reopen issue
                    </button>
                </div>
            </div>
            <div class="flex items-center gap-3 text-sm">
                <span v-if="issue.state === 'open'" class="bg-[#238636] text-white px-3 py-1 rounded-full flex items-center gap-1">
                     <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-issue-opened fill-current"><path d="M8 9.5a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3Z"></path><path d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0ZM1.5 8a6.5 6.5 0 1 0 13 0 6.5 6.5 0 0 0-13 0Z"></path></svg>
                     Open
                </span>
                <span v-else class="bg-[#8957e5] text-white px-3 py-1 rounded-full flex items-center gap-1">
                     <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-issue-closed fill-current"><path d="M11.28 6.78a.75.75 0 0 0-1.06-1.06L7.25 8.69 5.78 7.22a.75.75 0 0 0-1.06 1.06l2 2a.75.75 0 0 0 1.06 0l3.5-3.5Z"></path><path d="M16 8A8 8 0 1 1 0 8a8 8 0 0 1 16 0Zm-1.5 0a6.5 6.5 0 1 0-13 0 6.5 6.5 0 0 0 13 0Z"></path></svg>
                     Closed
                </span>
                <span class="text-gray-400">
                    <strong class="text-white">{{ issue.author_id }}</strong> opened this issue on {{ issue.created_at }} · {{ issue.comments }} comments
                </span>
            </div>
        </div>

        <!-- Discussion Timeline -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div class="col-span-3 space-y-6">
                <!-- Original Post -->
                <div class="flex gap-4">
                    <img :src="authorAvatar" class="w-10 h-10 rounded-full border border-gray-700" />
                    <div class="flex-grow border border-gray-700 rounded-md bg-[#161b22]">
                        <div class="bg-[#21262d] px-4 py-2 border-b border-gray-700 rounded-t-md text-sm flex justify-between">
                             <div class="text-gray-300"><strong>{{ issue.author_id }}</strong> commented on {{ issue.created_at }}</div>
                             <div class="text-gray-500">Owner</div>
                        </div>
                        <div class="p-4 text-gray-300">
                            {{ issue.body }}
                        </div>
                    </div>
                </div>

                <!-- Comments (Mock for loop) -->
                <div v-for="i in Math.min(issue.comments, 3)" :key="i" class="flex gap-4">
                     <img :src="getCommentUserAvatar(i)" class="w-10 h-10 rounded-full border border-gray-700" />
                     <div class="flex-grow border border-gray-700 rounded-md bg-[#161b22]">
                        <div class="bg-[#21262d] px-4 py-2 border-b border-gray-700 rounded-t-md text-sm">
                             <div class="text-gray-300"><strong>user_{{(i % 10) + 1}}</strong> commented</div>
                        </div>
                        <div class="p-4 text-gray-300">
                            This is a sample comment #{{i}} on this issue.
                        </div>
                     </div>
                </div>

                <hr class="border-gray-700" />

                <!-- New Comment Form -->
                <div class="flex gap-4">
                     <img src="/images/photo1764838101.jpg" class="w-10 h-10 rounded-full border border-gray-700" />
                     <div class="flex-grow space-y-2">
                         <textarea 
                           id="new-comment-textarea" 
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
                    <h3 class="text-gray-400 font-semibold mb-2 hover:text-blue-400 cursor-pointer">Assignees</h3>
                    <div class="text-gray-500">No one assigned</div>
                </div>
                <div class="border-b border-gray-700 pb-4">
                    <h3 class="text-gray-400 font-semibold mb-2 hover:text-blue-400 cursor-pointer">Labels</h3>
                    <div class="flex flex-wrap gap-2">
                        <span v-for="label in issue.labels" :key="label" class="px-2 py-0.5 rounded-full border border-gray-600 text-gray-300 bg-gray-800">
                            {{ label }}
                        </span>
                    </div>
                </div>
                <div class="border-b border-gray-700 pb-4">
                    <h3 class="text-gray-400 font-semibold mb-2 hover:text-blue-400 cursor-pointer">Projects</h3>
                    <div class="text-gray-500">None yet</div>
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
  name: 'ISSUE_DETAIL',
  setup() {
    const router = useRouter();
    const route = useRoute();
    const store = useSignatureStore();
    const dataStore = useDataStore();

    const newComment = ref('');

    const issue = computed(() => {
        const id = route.params.item_id || store.signature.issues_selected_issue_id;
        return dataStore.issues.find(i => i.id === id);
    });

    const author = computed(() => {
        if (!issue.value) return null;
        return dataStore.users.find(u => u.id === issue.value.author_id);
    });

    const authorAvatar = computed(() => {
        return author.value?.avatar || '/images/User.jpg';
    });

    const getCommentUserAvatar = (index) => {
        const userId = `user_${(index % 10) + 1}`;
        const user = dataStore.users.find(u => u.id === userId);
        return user?.avatar || '/images/User.jpg';
    };

    const goBack = async () => {
        store.setCurrentPageId('ISSUES_LIST');
        await router.push({ name: 'ISSUES_LIST' });
    };

    const closeIssue = () => {
        // Mock state update
        issue.value.state = 'closed';
    };

    const reopenIssue = () => {
        // Mock state update
        issue.value.state = 'open';
    };

    onMounted(() => {
        if (!issue.value) {
            // router.push({ name: 'ISSUES_LIST' });
        }
    });

    return {
        issue,
        authorAvatar,
        getCommentUserAvatar,
        newComment,
        goBack,
        closeIssue,
        reopenIssue
    };
  }
}
</script>