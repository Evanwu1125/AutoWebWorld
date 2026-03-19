import { createRouter, createWebHistory } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

// Page Imports
import HOME from '../pages/HOME.vue';
import REPOSITORIES_LIST from '../pages/REPOSITORIES_LIST.vue';
import REPOSITORY_DETAIL from '../pages/REPOSITORY_DETAIL.vue';
import NEW_REPOSITORY from '../pages/NEW_REPOSITORY.vue';
import REPO_CREATE_SUCCESS from '../pages/REPO_CREATE_SUCCESS.vue';
import ISSUES_LIST from '../pages/ISSUES_LIST.vue';
import ISSUE_DETAIL from '../pages/ISSUE_DETAIL.vue';
import NEW_ISSUE from '../pages/NEW_ISSUE.vue';
import ISSUE_CREATE_SUCCESS from '../pages/ISSUE_CREATE_SUCCESS.vue';
import PULL_REQUESTS_LIST from '../pages/PULL_REQUESTS_LIST.vue';
import PULL_REQUEST_DETAIL from '../pages/PULL_REQUEST_DETAIL.vue';
import NEW_PULL_REQUEST from '../pages/NEW_PULL_REQUEST.vue';
import PR_CREATE_SUCCESS from '../pages/PR_CREATE_SUCCESS.vue';
import BRANCHES_LIST from '../pages/BRANCHES_LIST.vue';
import NEW_BRANCH from '../pages/NEW_BRANCH.vue';
import NEW_BRANCH_SUCCESS from '../pages/NEW_BRANCH_SUCCESS.vue';
import COMPARE_BRANCHES from '../pages/COMPARE_BRANCHES.vue';
import PROFILE_OVERVIEW from '../pages/PROFILE_OVERVIEW.vue';
import PROFILE_SETTINGS from '../pages/PROFILE_SETTINGS.vue';
import PROFILE_UPDATE_SUCCESS from '../pages/PROFILE_UPDATE_SUCCESS.vue';
import PROFILE_FOLLOWERS from '../pages/PROFILE_FOLLOWERS.vue';

const routes = [
  { path: '/', name: 'HOME', component: HOME },
  { path: '/repos', name: 'REPOSITORIES_LIST', component: REPOSITORIES_LIST },
  { path: '/repo/:item_id?', name: 'REPOSITORY_DETAIL', component: REPOSITORY_DETAIL },
  { path: '/new', name: 'NEW_REPOSITORY', component: NEW_REPOSITORY },
  { path: '/new/success', name: 'REPO_CREATE_SUCCESS', component: REPO_CREATE_SUCCESS },
  { path: '/issues', name: 'ISSUES_LIST', component: ISSUES_LIST },
  { path: '/issue/:item_id?', name: 'ISSUE_DETAIL', component: ISSUE_DETAIL },
  { path: '/issues/new', name: 'NEW_ISSUE', component: NEW_ISSUE },
  { path: '/issues/new/success', name: 'ISSUE_CREATE_SUCCESS', component: ISSUE_CREATE_SUCCESS },
  { path: '/pulls', name: 'PULL_REQUESTS_LIST', component: PULL_REQUESTS_LIST },
  { path: '/pull/:item_id?', name: 'PULL_REQUEST_DETAIL', component: PULL_REQUEST_DETAIL },
  { path: '/pulls/new', name: 'NEW_PULL_REQUEST', component: NEW_PULL_REQUEST },
  { path: '/pulls/new/success', name: 'PR_CREATE_SUCCESS', component: PR_CREATE_SUCCESS },
  { path: '/branches', name: 'BRANCHES_LIST', component: BRANCHES_LIST },
  { path: '/branches/new', name: 'NEW_BRANCH', component: NEW_BRANCH },
  { path: '/branches/new/success', name: 'NEW_BRANCH_SUCCESS', component: NEW_BRANCH_SUCCESS },
  { path: '/compare', name: 'COMPARE_BRANCHES', component: COMPARE_BRANCHES },
  { path: '/profile', name: 'PROFILE_OVERVIEW', component: PROFILE_OVERVIEW },
  { path: '/profile/settings', name: 'PROFILE_SETTINGS', component: PROFILE_SETTINGS },
  { path: '/profile/updated', name: 'PROFILE_UPDATE_SUCCESS', component: PROFILE_UPDATE_SUCCESS },
  { path: '/profile/followers/:item_id?', name: 'PROFILE_FOLLOWERS', component: PROFILE_FOLLOWERS },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

router.beforeEach((to, from, next) => {
  const signatureStore = useSignatureStore();
  signatureStore.setCurrentPageId(to.name);
  next();
});

export default router;