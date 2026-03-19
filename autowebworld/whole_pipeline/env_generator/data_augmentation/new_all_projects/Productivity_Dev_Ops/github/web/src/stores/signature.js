import { defineStore } from 'pinia';
import { ref } from 'vue';
import fsmData from '../../fsm.json';

export const useSignatureStore = defineStore('signature', () => {
  // State
  const currentPageId = ref(fsmData.meta.initial_page_id);
  
  // Signature Fields (initialized from schema)
  // Using a reactive object to hold dynamic signature fields
  const signature = ref({
    // HOME
    current_user_id: null,
    cookie_consent_given: null,

    // REPOSITORIES_LIST
    location_permission_granted: null,
    repositories: [],
    repos_matched_repo_id: null,
    repos_selected_repo_id: null,
    repos_has_searched: null,
    repos_viewport_anchor_id: null,
    repos_list_filters_applied: null,

    // NEW_REPOSITORY
    new_repo_name: null,
    new_repo_description: null,
    new_repo_private: null,
    new_repo_readme_template: null,

    // REPO_CREATE_SUCCESS
    success_message: null,

    // ISSUES_LIST
    issues: [],
    issues_matched_issue_id: null,
    issues_selected_issue_id: null,
    issues_has_searched: null,
    issues_viewport_anchor_id: null,
    issues_list_filters_applied: null,

    // ISSUE_DETAIL
    // issues_selected_issue_id shared

    // NEW_ISSUE
    new_issue_title: null,
    new_issue_body: null,
    new_issue_labels: [],
    new_issue_assignee: null,

    // PULL_REQUESTS_LIST
    pulls: [],
    pulls_matched_pr_id: null,
    pulls_selected_pr_id: null,
    pulls_has_searched: null,
    pulls_viewport_anchor_id: null,
    pulls_list_filters_applied: null,

    // NEW_PULL_REQUEST
    new_pr_base_branch: null,
    new_pr_compare_branch: null,
    new_pr_title: null,
    new_pr_body: null,

    // BRANCHES_LIST
    branches: [],
    branches_selected_branch_name: null,

    // NEW_BRANCH
    new_branch_name: null,
    new_branch_source: null,

    // COMPARE_BRANCHES
    compare_base: null,
    compare_head: null,

    // PROFILE
    profile_username: null,
    profile_name: null,
    profile_bio: null,
    profile_location: null,
    profile_website: null,

    // PROFILE_FOLLOWERS
    followers: [],
    followers_viewport_anchor_id: null,
    followers_selected_user_id: null
  });

  // Actions
  function setCurrentPageId(id) {
    currentPageId.value = id;
  }

  return {
    currentPageId,
    signature,
    setCurrentPageId
  };
}, {
  persist: {
    storage: sessionStorage
  }
});