import { createRouter, createWebHistory } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

// Page Components
import HOME from '../pages/HOME.vue'
import REPO_LIST from '../pages/REPO_LIST.vue'
import REPO_DETAIL from '../pages/REPO_DETAIL.vue'
import CREATE_REPO_FORM from '../pages/CREATE_REPO_FORM.vue'
import CREATE_REPO_REVIEW from '../pages/CREATE_REPO_REVIEW.vue'
import CREATE_REPO_SUCCESS from '../pages/CREATE_REPO_SUCCESS.vue'
import PR_LIST from '../pages/PR_LIST.vue'
import PR_DETAIL from '../pages/PR_DETAIL.vue'
import CREATE_PR_FORM from '../pages/CREATE_PR_FORM.vue'
import CREATE_PR_REVIEW from '../pages/CREATE_PR_REVIEW.vue'
import CREATE_PR_SUCCESS from '../pages/CREATE_PR_SUCCESS.vue'
import MERGE_PR_FORM from '../pages/MERGE_PR_FORM.vue'
import MERGE_PR_REVIEW from '../pages/MERGE_PR_REVIEW.vue'
import MERGE_PR_SUCCESS from '../pages/MERGE_PR_SUCCESS.vue'
import PIPELINE_LIST from '../pages/PIPELINE_LIST.vue'
import PIPELINE_DETAIL from '../pages/PIPELINE_DETAIL.vue'
import PIPELINE_CONFIG_FORM from '../pages/PIPELINE_CONFIG_FORM.vue'
import PIPELINE_CONFIG_REVIEW from '../pages/PIPELINE_CONFIG_REVIEW.vue'
import CREATE_PIPELINE_SUCCESS from '../pages/CREATE_PIPELINE_SUCCESS.vue'
import WORKSPACE_MEMBERS from '../pages/WORKSPACE_MEMBERS.vue'
import INVITE_MEMBER_FORM from '../pages/INVITE_MEMBER_FORM.vue'
import INVITE_MEMBER_REVIEW from '../pages/INVITE_MEMBER_REVIEW.vue'
import INVITE_USER_SUCCESS from '../pages/INVITE_USER_SUCCESS.vue'
import REPO_PR_LIST from '../pages/REPO_PR_LIST.vue'
import REPO_PIPELINES from '../pages/REPO_PIPELINES.vue'
import REPO_SETTINGS from '../pages/REPO_SETTINGS.vue'

const routes = [
  { path: '/', name: 'HOME', component: HOME },
  { path: '/repositories', name: 'REPO_LIST', component: REPO_LIST },
  { path: '/repositories/:repo_id', name: 'REPO_DETAIL', component: REPO_DETAIL },
  { path: '/create-repo', name: 'CREATE_REPO_FORM', component: CREATE_REPO_FORM },
  { path: '/create-repo/review', name: 'CREATE_REPO_REVIEW', component: CREATE_REPO_REVIEW },
  { path: '/create-repo/success', name: 'CREATE_REPO_SUCCESS', component: CREATE_REPO_SUCCESS },
  { path: '/pull-requests', name: 'PR_LIST', component: PR_LIST },
  { path: '/pull-requests/:pr_id', name: 'PR_DETAIL', component: PR_DETAIL },
  { path: '/create-pr', name: 'CREATE_PR_FORM', component: CREATE_PR_FORM },
  { path: '/create-pr/review', name: 'CREATE_PR_REVIEW', component: CREATE_PR_REVIEW },
  { path: '/create-pr/success', name: 'CREATE_PR_SUCCESS', component: CREATE_PR_SUCCESS },
  { path: '/pull-requests/:pr_id/merge', name: 'MERGE_PR_FORM', component: MERGE_PR_FORM },
  { path: '/pull-requests/:pr_id/merge/review', name: 'MERGE_PR_REVIEW', component: MERGE_PR_REVIEW },
  { path: '/pull-requests/:pr_id/merge/success', name: 'MERGE_PR_SUCCESS', component: MERGE_PR_SUCCESS },
  { path: '/pipelines', name: 'PIPELINE_LIST', component: PIPELINE_LIST },
  { path: '/pipelines/:pipeline_id', name: 'PIPELINE_DETAIL', component: PIPELINE_DETAIL },
  { path: '/pipelines/config', name: 'PIPELINE_CONFIG_FORM', component: PIPELINE_CONFIG_FORM },
  { path: '/pipelines/config/review', name: 'PIPELINE_CONFIG_REVIEW', component: PIPELINE_CONFIG_REVIEW },
  { path: '/pipelines/create/success', name: 'CREATE_PIPELINE_SUCCESS', component: CREATE_PIPELINE_SUCCESS },
  { path: '/members', name: 'WORKSPACE_MEMBERS', component: WORKSPACE_MEMBERS },
  { path: '/members/invite', name: 'INVITE_MEMBER_FORM', component: INVITE_MEMBER_FORM },
  { path: '/members/invite/review', name: 'INVITE_MEMBER_REVIEW', component: INVITE_MEMBER_REVIEW },
  { path: '/members/invite/success', name: 'INVITE_USER_SUCCESS', component: INVITE_USER_SUCCESS },
  { path: '/repositories/:repo_id/pull-requests', name: 'REPO_PR_LIST', component: REPO_PR_LIST },
  { path: '/repositories/:repo_id/pipelines', name: 'REPO_PIPELINES', component: REPO_PIPELINES },
  { path: '/repositories/:repo_id/settings', name: 'REPO_SETTINGS', component: REPO_SETTINGS },
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

router.beforeEach((to, from, next) => {
  const signatureStore = useSignatureStore()
  signatureStore.setCurrentPageId(to.name)
  next()
})

export default router