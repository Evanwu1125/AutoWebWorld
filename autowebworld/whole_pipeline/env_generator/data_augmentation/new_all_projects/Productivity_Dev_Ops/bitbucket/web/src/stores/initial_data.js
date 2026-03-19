import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useDataStore = defineStore('data', () => {
  // Repositories
  const repositories = ref([
    { id: 'repo_001', name: 'frontend-core', owner: 'Team Alpha', access: 'private', description: 'Core frontend library', updated_at: '2023-10-01', activity: 85, image: '/images/photo1765609035.jpg' },
    { id: 'repo_002', name: 'backend-api', owner: 'Team Beta', access: 'private', description: 'Main API service', updated_at: '2023-10-02', activity: 92, image: '/images/BackendAPI.jpg' },
    { id: 'repo_003', name: 'design-system', owner: 'Design Team', access: 'public', description: 'UI component library', updated_at: '2023-09-28', activity: 40, image: '/images/DesignSystem.jpg' },
    { id: 'repo_004', name: 'mobile-app', owner: 'Team Gamma', access: 'private', description: 'iOS and Android app', updated_at: '2023-10-03', activity: 75, image: '/images/MobileAppDevelopment.jpg' },
    { id: 'repo_005', name: 'analytics-service', owner: 'Data Team', access: 'private', description: 'Data processing engine', updated_at: '2023-09-25', activity: 60, image: '/images/DataAnalytics.jpg' },
    { id: 'repo_006', name: 'auth-provider', owner: 'Security Team', access: 'private', description: 'OAuth2 provider', updated_at: '2023-10-01', activity: 30, image: '/images/security-lock-auth.jpg' },
    { id: 'repo_007', name: 'payment-gateway', owner: 'Team Alpha', access: 'private', description: 'Payment processing', updated_at: '2023-09-30', activity: 88, image: '/images/photo1765609035.jpg' },
    { id: 'repo_008', name: 'notification-center', owner: 'Team Beta', access: 'public', description: 'Email and Push notifications', updated_at: '2023-09-20', activity: 20, image: '/images/photo1765609035.jpg' },
    { id: 'repo_009', name: 'infrastructure-iac', owner: 'DevOps', access: 'private', description: 'Terraform scripts', updated_at: '2023-10-03', activity: 50, image: '/images/CloudInfrastructure.jpg' },
    { id: 'repo_010', name: 'landing-page', owner: 'Marketing', access: 'public', description: 'Corporate website', updated_at: '2023-10-02', activity: 45, image: '/images/photo1765609034.jpg' },
    { id: 'repo_011', name: 'documentation', owner: 'Tech Writers', access: 'public', description: 'Product docs', updated_at: '2023-09-15', activity: 10, image: '/images/photo1765609035.jpg' },
    { id: 'repo_012', name: 'legacy-monolith', owner: 'Old Guard', access: 'private', description: 'Do not touch', updated_at: '2023-01-01', activity: 5, image: '/images/LegacyServer.jpg' },
    { id: 'repo_013', name: 'micro-frontend-1', owner: 'Team Alpha', access: 'private', description: 'Dashboard module', updated_at: '2023-10-01', activity: 70, image: '/images/Dashboard.jpg' },
    { id: 'repo_014', name: 'micro-frontend-2', owner: 'Team Beta', access: 'private', description: 'Settings module', updated_at: '2023-09-29', activity: 65, image: '/images/photo1765609035.jpg' },
    { id: 'repo_015', name: 'utils-lib', owner: 'Team Gamma', access: 'public', description: 'Common utilities', updated_at: '2023-09-22', activity: 25, image: '/images/photo1765609035.jpg' },
    { id: 'repo_016', name: 'cli-tools', owner: 'DevOps', access: 'public', description: 'Developer CLI', updated_at: '2023-09-18', activity: 55, image: '/images/CLI.jpg' }
  ])

  // Pull Requests
  const pull_requests = ref([
    { id: 'pr_001', title: 'Fix login bug', author_id: 'user_001', repo_id: 'repo_006', status: 'open', created_at: '2023-10-01', updated_at: '2023-10-03', image: '/images/CodeReview.jpg' },
    { id: 'pr_002', title: 'Add dark mode', author_id: 'user_002', repo_id: 'repo_003', status: 'merged', created_at: '2023-09-28', updated_at: '2023-09-30', image: '/images/photo1765609035.jpg' },
    { id: 'pr_003', title: 'Update dependencies', author_id: 'user_003', repo_id: 'repo_002', status: 'open', created_at: '2023-10-02', updated_at: '2023-10-02', image: '/images/photo1765609034.jpg' },
    { id: 'pr_004', title: 'Refactor user service', author_id: 'user_001', repo_id: 'repo_002', status: 'declined', created_at: '2023-09-25', updated_at: '2023-09-26', image: '/images/CodeRefactoring.jpg' },
    { id: 'pr_005', title: 'Implement search', author_id: 'user_004', repo_id: 'repo_001', status: 'open', created_at: '2023-10-03', updated_at: '2023-10-03', image: '/images/photo1765609035.jpg' },
    { id: 'pr_006', title: 'Fix typo in docs', author_id: 'user_005', repo_id: 'repo_011', status: 'merged', created_at: '2023-09-15', updated_at: '2023-09-16', image: '/images/Typo.jpg' },
    { id: 'pr_007', title: 'Add payment method', author_id: 'user_002', repo_id: 'repo_007', status: 'open', created_at: '2023-10-01', updated_at: '2023-10-02', image: '/images/photo1765609035.jpg' },
    { id: 'pr_008', title: 'Optimize images', author_id: 'user_003', repo_id: 'repo_010', status: 'open', created_at: '2023-10-02', updated_at: '2023-10-03', image: '/images/ImageOptimization.jpg' },
    { id: 'pr_009', title: 'Add unit tests', author_id: 'user_001', repo_id: 'repo_001', status: 'merged', created_at: '2023-09-20', updated_at: '2023-09-22', image: '/images/UnitTesting.jpg' },
    { id: 'pr_010', title: 'Setup CI/CD', author_id: 'user_004', repo_id: 'repo_009', status: 'merged', created_at: '2023-09-10', updated_at: '2023-09-12', image: '/images/ci-cd-pipeline.jpg' },
    { id: 'pr_011', title: 'Create new endpoint', author_id: 'user_002', repo_id: 'repo_002', status: 'open', created_at: '2023-10-03', updated_at: '2023-10-03', image: '/images/photo1765609035.jpg' },
    { id: 'pr_012', title: 'Update logo', author_id: 'user_005', repo_id: 'repo_003', status: 'open', created_at: '2023-10-01', updated_at: '2023-10-01', image: '/images/Logo.jpg' },
    { id: 'pr_013', title: 'Fix crash on iOS', author_id: 'user_003', repo_id: 'repo_004', status: 'open', created_at: '2023-09-29', updated_at: '2023-10-01', image: '/images/iOSCrashFix.jpg' },
    { id: 'pr_014', title: 'Add analytics events', author_id: 'user_001', repo_id: 'repo_004', status: 'open', created_at: '2023-10-02', updated_at: '2023-10-03', image: '/images/photo1765609035.jpg' },
    { id: 'pr_015', title: 'Remove deprecated code', author_id: 'user_004', repo_id: 'repo_012', status: 'declined', created_at: '2023-09-01', updated_at: '2023-09-02', image: '/images/DeprecatedCode.jpg' }
  ])

  // Pipelines
  const pipelines = ref([
    { id: 'pipe_001', name: 'Build & Test', repo_id: 'repo_001', status: 'success', branch: 'main', trigger: 'on_push', created_at: '2023-10-03 10:00', image: '/images/photo1765609035.jpg' },
    { id: 'pipe_002', name: 'Deploy to Staging', repo_id: 'repo_001', status: 'running', branch: 'develop', trigger: 'on_push', created_at: '2023-10-03 10:15', image: '/images/photo1765609035.jpg' },
    { id: 'pipe_003', name: 'Deploy to Prod', repo_id: 'repo_001', status: 'failed', branch: 'main', trigger: 'manual', created_at: '2023-10-02 14:00', image: '/images/ProductionServer.jpg' },
    { id: 'pipe_004', name: 'Lint Check', repo_id: 'repo_002', status: 'success', branch: 'feature/login', trigger: 'on_push', created_at: '2023-10-03 09:30', image: '/images/CodeLinting.jpg' },
    { id: 'pipe_005', name: 'Integration Tests', repo_id: 'repo_002', status: 'success', branch: 'main', trigger: 'schedule', created_at: '2023-10-03 00:00', image: '/images/IntegrationTesting.jpg' },
    { id: 'pipe_006', name: 'Build Android', repo_id: 'repo_004', status: 'running', branch: 'develop', trigger: 'on_push', created_at: '2023-10-03 11:00', image: '/images/AndroidBuild.jpg' },
    { id: 'pipe_007', name: 'Build iOS', repo_id: 'repo_004', status: 'success', branch: 'develop', trigger: 'on_push', created_at: '2023-10-03 11:00', image: '/images/photo1765609035.jpg' },
    { id: 'pipe_008', name: 'Security Scan', repo_id: 'repo_006', status: 'success', branch: 'main', trigger: 'schedule', created_at: '2023-10-01 00:00', image: '/images/Security.jpg' },
    { id: 'pipe_009', name: 'Publish Docs', repo_id: 'repo_011', status: 'success', branch: 'main', trigger: 'on_push', created_at: '2023-09-15 12:00', image: '/images/photo1765609034.jpg' },
    { id: 'pipe_010', name: 'Terraform Plan', repo_id: 'repo_009', status: 'success', branch: 'main', trigger: 'manual', created_at: '2023-10-03 08:00', image: '/images/photo1765609035.jpg' },
    { id: 'pipe_011', name: 'Terraform Apply', repo_id: 'repo_009', status: 'failed', branch: 'main', trigger: 'manual', created_at: '2023-10-03 08:15', image: '/images/photo1765609034.jpg' },
    { id: 'pipe_012', name: 'Unit Tests', repo_id: 'repo_003', status: 'success', branch: 'main', trigger: 'on_push', created_at: '2023-09-28 15:00', image: '/images/photo1765609035.jpg' },
    { id: 'pipe_013', name: 'E2E Tests', repo_id: 'repo_007', status: 'success', branch: 'develop', trigger: 'schedule', created_at: '2023-10-03 02:00', image: '/images/E2ETesting.jpg' },
    { id: 'pipe_014', name: 'Release', repo_id: 'repo_016', status: 'success', branch: 'main', trigger: 'manual', created_at: '2023-09-18 10:00', image: '/images/photo1765609034.jpg' },
    { id: 'pipe_015', name: 'Nightly Build', repo_id: 'repo_001', status: 'success', branch: 'develop', trigger: 'schedule', created_at: '2023-10-03 03:00', image: '/images/photo1765609035.jpg' }
  ])

  // Workspace Members
  const members = ref([
    { id: 'user_001', name: 'Alice Smith', role: 'admin', email: 'alice@example.com', active: 95, image: '/images/Profile.jpg' },
    { id: 'user_002', name: 'Bob Jones', role: 'developer', email: 'bob@example.com', active: 80, image: '/images/Profile.jpg' },
    { id: 'user_003', name: 'Charlie Brown', role: 'developer', email: 'charlie@example.com', active: 60, image: '/images/photo1765609035.jpg' },
    { id: 'user_004', name: 'David Wilson', role: 'viewer', email: 'david@example.com', active: 20, image: '/images/Businessman.jpg' },
    { id: 'user_005', name: 'Eva Green', role: 'admin', email: 'eva@example.com', active: 90, image: '/images/BusinessWoman.jpg' },
    { id: 'user_006', name: 'Frank White', role: 'developer', email: 'frank@example.com', active: 75, image: '/images/Businessman.jpg' },
    { id: 'user_007', name: 'Grace Lee', role: 'viewer', email: 'grace@example.com', active: 10, image: '/images/Glasses.jpg' },
    { id: 'user_008', name: 'Henry Ford', role: 'developer', email: 'henry@example.com', active: 50, image: '/images/photo1765609034.jpg' },
    { id: 'user_009', name: 'Ivy Chen', role: 'developer', email: 'ivy@example.com', active: 85, image: '/images/photo1765609035.jpg' },
    { id: 'user_010', name: 'Jack Black', role: 'admin', email: 'jack@example.com', active: 100, image: '/images/photo1765609035.jpg' },
    { id: 'user_011', name: 'Kelly Clarkson', role: 'viewer', email: 'kelly@example.com', active: 5, image: '/images/Profile.jpg' },
    { id: 'user_012', name: 'Leo Messi', role: 'developer', email: 'leo@example.com', active: 92, image: '/images/photo1765609035.jpg' }
  ])

  return {
    repositories,
    pull_requests,
    pipelines,
    members
  }
}, {
  persist: {
    storage: sessionStorage
  }
})