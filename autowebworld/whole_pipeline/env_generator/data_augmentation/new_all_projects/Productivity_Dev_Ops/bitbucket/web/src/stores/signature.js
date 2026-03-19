import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useSignatureStore = defineStore('signature', () => {
  // Global Signature Schema
  const location_permission_granted = ref(null)
  const current_user_id = ref('user_001') // Default user
  const repositories = ref([])
  const pull_requests = ref([])
  const pipelines = ref([])
  const workspace_members = ref([])
  const cookie_consent_given = ref(null)

  // Page Specific Signature Fields (Union of all page schemas)
  // HOME
  // REPO_LIST
  const repo_list_has_searched = ref(null)
  const matched_repo_id = ref(null)
  const repo_list_viewport_anchor_id = ref(null)
  const selected_repo_id = ref(null)
  const repo_list_filters_applied = ref(null)
  
  // REPO_DETAIL
  const selected_pipeline_id = ref(null)

  // CREATE_REPO_FORM
  const repo_name = ref(null)
  const repo_access_level = ref(null)
  const repo_description = ref(null)

  // CREATE_REPO_SUCCESS
  const success_message = ref(null)

  // PR_LIST
  const pr_list_has_searched = ref(null)
  const matched_pr_id = ref(null)
  const pr_list_viewport_anchor_id = ref(null)
  const selected_pr_id = ref(null)
  const pr_list_filters_applied = ref(null)

  // PR_DETAIL
  const merge_commit_message = ref(null)

  // CREATE_PR_FORM
  const pr_title = ref(null)
  const pr_description = ref(null)
  const pr_source_branch = ref(null)
  const pr_target_branch = ref(null)

  // MERGE_PR_FORM
  const merge_strategy = ref(null)

  // PIPELINE_LIST
  const pipeline_list_filters_applied = ref(null)
  const pipeline_list_has_searched = ref(null)
  const matched_pipeline_id = ref(null)
  const pipeline_list_viewport_anchor_id = ref(null)
  // selected_pipeline_id already defined

  // PIPELINE_CONFIG_FORM
  const pipeline_name = ref(null)
  const pipeline_trigger = ref(null)
  const pipeline_branch = ref(null)

  // WORKSPACE_MEMBERS
  const members_filters_applied = ref(null)
  const members_has_searched = ref(null)
  const matched_member_id = ref(null)
  const members_viewport_anchor_id = ref(null)
  const selected_member_id = ref(null)

  // INVITE_MEMBER_FORM
  const invite_email = ref(null)
  const invite_role = ref(null)

  // REPO_PR_LIST
  const repo_pr_list_has_searched = ref(null)
  const matched_repo_pr_id = ref(null)
  const repo_pr_list_viewport_anchor_id = ref(null)
  // selected_pr_id already defined

  // REPO_PIPELINES
  const repo_pipelines_viewport_anchor_id = ref(null)

  // REPO_SETTINGS
  const default_branch = ref(null)

  // Internal State
  const currentPageId = ref('HOME')

  function setCurrentPageId(pageId) {
    currentPageId.value = pageId
  }

  return {
    location_permission_granted,
    current_user_id,
    repositories,
    pull_requests,
    pipelines,
    workspace_members,
    cookie_consent_given,
    
    repo_list_has_searched,
    matched_repo_id,
    repo_list_viewport_anchor_id,
    selected_repo_id,
    repo_list_filters_applied,
    
    selected_pipeline_id,
    
    repo_name,
    repo_access_level,
    repo_description,
    
    success_message,
    
    pr_list_has_searched,
    matched_pr_id,
    pr_list_viewport_anchor_id,
    selected_pr_id,
    pr_list_filters_applied,
    
    merge_commit_message,
    
    pr_title,
    pr_description,
    pr_source_branch,
    pr_target_branch,
    
    merge_strategy,
    
    pipeline_list_filters_applied,
    pipeline_list_has_searched,
    matched_pipeline_id,
    pipeline_list_viewport_anchor_id,
    
    pipeline_name,
    pipeline_trigger,
    pipeline_branch,
    
    members_filters_applied,
    members_has_searched,
    matched_member_id,
    members_viewport_anchor_id,
    selected_member_id,
    
    invite_email,
    invite_role,
    
    repo_pr_list_has_searched,
    matched_repo_pr_id,
    repo_pr_list_viewport_anchor_id,
    
    repo_pipelines_viewport_anchor_id,
    
    default_branch,
    
    currentPageId,
    setCurrentPageId
  }
}, {
  persist: {
    storage: sessionStorage
  }
})