import { defineStore } from 'pinia';
import { ref } from 'vue';

export const useDataStore = defineStore('data', () => {
  // --- Mock Data ---
  
  // Users
  const users = ref([
    { id: 'user_1', username: 'octocat', name: 'The Octocat', avatar: '/images/User.jpg', bio: 'GitHub Mascot', location: 'San Francisco', website: 'https://github.com' },
    { id: 'user_2', username: 'hubot', name: 'Hubot', avatar: '/images/user-2.jpg', bio: 'GitHub Bot', location: 'The Cloud', website: 'https://hubot.github.com' },
    { id: 'user_3', username: 'torvalds', name: 'Linus Torvalds', avatar: '/images/LinusTorvalds.jpg', bio: 'Linux Creator', location: 'Portland, OR', website: '' },
    { id: 'user_4', username: 'defunkt', name: 'Chris Wanstrath', avatar: '/images/GitHub.jpg', bio: 'GitHub Co-founder', location: 'San Francisco', website: '' },
    { id: 'user_5', username: 'mojombo', name: 'Tom Preston-Werner', avatar: '/images/GitHubCofounder.jpg', bio: 'GitHub Co-founder', location: 'San Francisco', website: '' },
    { id: 'user_6', username: 'wycats', name: 'Yehuda Katz', avatar: '/images/YehudaKatz.jpg', bio: 'Rust, Ember, Rails', location: 'Portland', website: '' },
    { id: 'user_7', username: 'ezmobius', name: 'Ezra Zygmuntowicz', avatar: '/images/Rubyist.jpg', bio: 'Rubyist', location: 'In memory', website: '' },
    { id: 'user_8', username: 'ivey', name: 'Michael Ivey', avatar: '/images/Developer.jpg', bio: 'Developer', location: 'Alabama', website: '' },
    { id: 'user_9', username: 'evanphx', name: 'Evan Phoenix', avatar: '/images/User.jpg', bio: 'Rubinius', location: 'Los Angeles', website: '' },
    { id: 'user_10', username: 'vanpelt', name: 'Chris Van Pelt', avatar: '/images/User.jpg', bio: 'CrowdFlower', location: 'San Francisco', website: '' }
  ]);

  // Repositories
  const repositories = ref([
    { id: "repo_1", name: "sample-repo", description: "A sample repository for testing", private: false, stars: 1200, updated_at: "2023-10-01", language: "JavaScript", owner_id: "user_1", image: "/images/repo-1.jpg" },
    { id: "repo_2", name: "linux", description: "Linux kernel source tree", private: false, stars: 150000, updated_at: "2023-11-15", language: "C", owner_id: "user_3", image: "/images/repo-2.jpg" },
    { id: "repo_3", name: "git", description: "Git source code", private: false, stars: 45000, updated_at: "2023-11-10", language: "C", owner_id: "user_1", image: "/images/repo-3.jpg" },
    { id: "repo_4", name: "vue", description: "Vue.js Core", private: false, stars: 200000, updated_at: "2023-11-18", language: "TypeScript", owner_id: "user_6", image: "/images/repo-4.jpg" },
    { id: "repo_5", name: "react", description: "A declarative, efficient, and flexible JavaScript library for building user interfaces.", private: false, stars: 210000, updated_at: "2023-11-19", language: "JavaScript", owner_id: "user_1", image: "/images/repo-5.jpg" },
    { id: "repo_6", name: "vscode", description: "Visual Studio Code", private: false, stars: 140000, updated_at: "2023-11-20", language: "TypeScript", owner_id: "user_1", image: "/images/repo-6.jpg" },
    { id: "repo_7", name: "bootstrap", description: "The most popular HTML, CSS, and JS library in the world.", private: false, stars: 160000, updated_at: "2023-10-25", language: "CSS", owner_id: "user_5", image: "/images/repo-7.jpg" },
    { id: "repo_8", name: "tensorflow", description: "Computation using data flow graphs for scalable machine learning", private: false, stars: 170000, updated_at: "2023-11-12", language: "C++", owner_id: "user_1", image: "/images/repo-8.jpg" },
    { id: "repo_9", name: "private-project", description: "Top secret project", private: true, stars: 5, updated_at: "2023-11-21", language: "Python", owner_id: "user_1", image: "/images/repo-9.jpg" },
    { id: "repo_10", name: "node", description: "Node.js JavaScript runtime", private: false, stars: 98000, updated_at: "2023-11-14", language: "JavaScript", owner_id: "user_1", image: "/images/repo-10.jpg" },
    { id: "repo_11", name: "electron", description: "Build cross-platform desktop apps with JavaScript, HTML, and CSS", private: false, stars: 105000, updated_at: "2023-11-01", language: "C++", owner_id: "user_1", image: "/images/repo-11.jpg" },
    { id: "repo_12", name: "angular", description: "One framework. Mobile & desktop.", private: false, stars: 85000, updated_at: "2023-10-30", language: "TypeScript", owner_id: "user_1", image: "/images/repo-12.jpg" },
    { id: "repo_13", name: "flutter", description: "Flutter makes it easy and fast to build beautiful apps for mobile and beyond", private: false, stars: 155000, updated_at: "2023-11-22", language: "Dart", owner_id: "user_1", image: "/images/repo-13.jpg" },
    { id: "repo_14", name: "d3", description: "Bring data to life with SVG, Canvas and HTML.", private: false, stars: 100000, updated_at: "2023-09-15", language: "JavaScript", owner_id: "user_5", image: "/images/repo-14.jpg" },
    { id: "repo_15", name: "django", description: "The Web framework for perfectionists with deadlines.", private: false, stars: 70000, updated_at: "2023-11-05", language: "Python", owner_id: "user_1", image: "/images/repo-15.jpg" },
    { id: "repo_16", name: "rails", description: "Ruby on Rails", private: false, stars: 53000, updated_at: "2023-10-20", language: "Ruby", owner_id: "user_4", image: "/images/repo-16.jpg" },
    { id: "repo_17", name: "laravel", description: "A PHP framework for web artisans", private: false, stars: 73000, updated_at: "2023-11-11", language: "PHP", owner_id: "user_1", image: "/images/repo-17.jpg" },
    { id: "repo_18", name: "webpack", description: "A bundler for javascript and friends.", private: false, stars: 63000, updated_at: "2023-08-10", language: "JavaScript", owner_id: "user_1", image: "/images/repo-18.jpg" },
    { id: "repo_19", name: "kubernetes", description: "Production-Grade Container Scheduling and Management", private: false, stars: 102000, updated_at: "2023-11-17", language: "Go", owner_id: "user_1", image: "/images/repo-19.jpg" },
    { id: "repo_20", name: "ansible", description: "Ansible is a radically simple IT automation platform", private: false, stars: 58000, updated_at: "2023-10-12", language: "Python", owner_id: "user_1", image: "/images/repo-20.jpg" }
  ]);

  // Issues
  const issues = ref([
    { id: "issue_1", repo_id: "repo_1", title: "Bug report: App crashes on start", body: "Steps to reproduce...", state: "open", comments: 5, created_at: "2023-11-20", author_id: "user_2", labels: ["bug"] },
    { id: "issue_2", repo_id: "repo_1", title: "Feature request: Dark mode", body: "Please add dark mode", state: "open", comments: 12, created_at: "2023-11-18", author_id: "user_3", labels: ["enhancement"] },
    { id: "issue_3", repo_id: "repo_1", title: "Documentation update needed", body: "Readme is outdated", state: "closed", comments: 2, created_at: "2023-10-05", author_id: "user_4", labels: ["documentation"] },
    { id: "issue_4", repo_id: "repo_2", title: "Kernel panic on boot", body: "Log attached", state: "open", comments: 50, created_at: "2023-11-21", author_id: "user_5", labels: ["bug", "critical"] },
    { id: "issue_5", repo_id: "repo_4", title: "Composition API question", body: "How to use with...", state: "open", comments: 8, created_at: "2023-11-15", author_id: "user_7", labels: ["question"] },
    { id: "issue_6", repo_id: "repo_1", title: "Typo in main.js", body: "Line 42", state: "open", comments: 1, created_at: "2023-11-22", author_id: "user_8", labels: ["bug", "good first issue"] },
    { id: "issue_7", repo_id: "repo_6", title: "Extension API not responding", body: "Calling command...", state: "open", comments: 3, created_at: "2023-11-19", author_id: "user_9", labels: ["bug"] },
    { id: "issue_8", repo_id: "repo_5", title: "React Hooks performace", body: "UseMemo not working as expected", state: "closed", comments: 15, created_at: "2023-09-10", author_id: "user_10", labels: ["bug"] },
    { id: "issue_9", repo_id: "repo_1", title: "Add support for mobile", body: "Responsive design", state: "open", comments: 6, created_at: "2023-11-10", author_id: "user_2", labels: ["enhancement"] },
    { id: "issue_10", repo_id: "repo_3", title: "Merge conflict resolution", body: "Algorithm improvement", state: "open", comments: 20, created_at: "2023-11-01", author_id: "user_3", labels: ["enhancement"] },
    { id: "issue_11", repo_id: "repo_1", title: "Login button not working", body: "Nothing happens on click", state: "open", comments: 4, created_at: "2023-11-23", author_id: "user_4", labels: ["bug"] },
    { id: "issue_12", repo_id: "repo_1", title: "Refactor component structure", body: "Too complex", state: "open", comments: 7, created_at: "2023-11-12", author_id: "user_5", labels: ["enhancement"] },
    { id: "issue_13", repo_id: "repo_8", title: "GPU memory leak", body: "OOM error on training", state: "open", comments: 30, created_at: "2023-11-16", author_id: "user_6", labels: ["bug"] },
    { id: "issue_14", repo_id: "repo_1", title: "Update dependencies", body: "Security vulnerabilities", state: "closed", comments: 2, created_at: "2023-10-20", author_id: "user_7", labels: ["maintenance"] },
    { id: "issue_15", repo_id: "repo_12", title: "Ivy compiler error", body: "Stack trace below", state: "open", comments: 9, created_at: "2023-11-14", author_id: "user_8", labels: ["bug"] }
  ]);

  // Pull Requests
  const pulls = ref([
    { id: "pr_1", repo_id: "repo_1", title: "Add new feature", body: "Implements feature X", state: "open", reviews: 2, created_at: "2023-11-21", author_id: "user_2", base: "main", head: "feature-x" },
    { id: "pr_2", repo_id: "repo_1", title: "Fix bug in login", body: "Corrects validation logic", state: "open", reviews: 5, created_at: "2023-11-20", author_id: "user_3", base: "main", head: "fix-login" },
    { id: "pr_3", repo_id: "repo_1", title: "Update README", body: "Adds installation steps", state: "merged", reviews: 1, created_at: "2023-11-15", author_id: "user_4", base: "main", head: "docs-update" },
    { id: "pr_4", repo_id: "repo_2", title: "New driver support", body: "Adds driver for Device Y", state: "open", reviews: 10, created_at: "2023-11-19", author_id: "user_5", base: "master", head: "driver-dev" },
    { id: "pr_5", repo_id: "repo_4", title: "Refactor v-model", body: "Internal cleanup", state: "open", reviews: 8, created_at: "2023-11-18", author_id: "user_6", base: "dev", head: "refactor" },
    { id: "pr_6", repo_id: "repo_1", title: "Add unit tests", body: "Coverage up to 80%", state: "open", reviews: 3, created_at: "2023-11-22", author_id: "user_7", base: "main", head: "tests" },
    { id: "pr_7", repo_id: "repo_6", title: "Language server update", body: "Bumps version", state: "closed", reviews: 0, created_at: "2023-11-10", author_id: "user_8", base: "main", head: "lsp-bump" },
    { id: "pr_8", repo_id: "repo_1", title: "Optimize images", body: "Compress assets", state: "open", reviews: 1, created_at: "2023-11-23", author_id: "user_9", base: "main", head: "assets" },
    { id: "pr_9", repo_id: "repo_1", title: "Change color scheme", body: "Darker blue", state: "open", reviews: 4, created_at: "2023-11-17", author_id: "user_10", base: "main", head: "design" },
    { id: "pr_10", repo_id: "repo_3", title: "Fix rebase crash", body: "Edge case handling", state: "open", reviews: 15, created_at: "2023-11-16", author_id: "user_2", base: "master", head: "rebase-fix" },
    { id: "pr_11", repo_id: "repo_1", title: "Implement search", body: "Elasticsearch integration", state: "open", reviews: 6, created_at: "2023-11-14", author_id: "user_3", base: "main", head: "search" },
    { id: "pr_12", repo_id: "repo_1", title: "Remove legacy code", body: "Cleanup", state: "merged", reviews: 2, created_at: "2023-11-05", author_id: "user_4", base: "main", head: "cleanup" },
    { id: "pr_13", repo_id: "repo_1", title: "Typo fix", body: "Small fix", state: "open", reviews: 0, created_at: "2023-11-24", author_id: "user_5", base: "main", head: "patch-1" },
    { id: "pr_14", repo_id: "repo_9", title: "Secret algorithm", body: "Optimization", state: "open", reviews: 1, created_at: "2023-11-20", author_id: "user_6", base: "main", head: "algo" },
    { id: "pr_15", repo_id: "repo_1", title: "Accessibility improvements", body: "ARIA labels", state: "open", reviews: 3, created_at: "2023-11-13", author_id: "user_7", base: "main", head: "a11y" }
  ]);

  // Branches
  const branches = ref([
    { id: "br_1", repo_id: "repo_1", name: "main", protected: true },
    { id: "br_2", repo_id: "repo_1", name: "develop", protected: false },
    { id: "br_3", repo_id: "repo_1", name: "feature-x", protected: false },
    { id: "br_4", repo_id: "repo_1", name: "fix-login", protected: false },
    { id: "br_5", repo_id: "repo_1", name: "docs-update", protected: false },
    { id: "br_6", repo_id: "repo_1", name: "tests", protected: false },
    { id: "br_7", repo_id: "repo_1", name: "assets", protected: false },
    { id: "br_8", repo_id: "repo_1", name: "design", protected: false },
    { id: "br_9", repo_id: "repo_1", name: "search", protected: false },
    { id: "br_10", repo_id: "repo_1", name: "cleanup", protected: false },
    { id: "br_11", repo_id: "repo_1", name: "patch-1", protected: false },
    { id: "br_12", repo_id: "repo_1", name: "a11y", protected: false },
    { id: "br_13", repo_id: "repo_1", name: "staging", protected: false },
    { id: "br_14", repo_id: "repo_1", name: "release-1.0", protected: true },
    { id: "br_15", repo_id: "repo_1", name: "experiment", protected: false }
  ]);

  // Followers (Mock relationships)
  const followers = ref([
    { id: "follow_1", user_id: "user_1", follower_id: "user_2" },
    { id: "follow_2", user_id: "user_1", follower_id: "user_3" },
    { id: "follow_3", user_id: "user_1", follower_id: "user_4" },
    { id: "follow_4", user_id: "user_1", follower_id: "user_5" },
    { id: "follow_5", user_id: "user_1", follower_id: "user_6" },
    { id: "follow_6", user_id: "user_1", follower_id: "user_7" },
    { id: "follow_7", user_id: "user_1", follower_id: "user_8" },
    { id: "follow_8", user_id: "user_1", follower_id: "user_9" },
    { id: "follow_9", user_id: "user_1", follower_id: "user_10" },
    { id: "follow_10", user_id: "user_2", follower_id: "user_1" },
    { id: "follow_11", user_id: "user_2", follower_id: "user_3" },
    { id: "follow_12", user_id: "user_3", follower_id: "user_1" },
    { id: "follow_13", user_id: "user_4", follower_id: "user_1" },
    { id: "follow_14", user_id: "user_5", follower_id: "user_1" },
    { id: "follow_15", user_id: "user_6", follower_id: "user_1" }
  ]);

  return {
    users,
    repositories,
    issues,
    pulls,
    branches,
    followers
  };
}, {
  persist: {
    storage: sessionStorage
  }
});