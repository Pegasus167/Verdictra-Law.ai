import { Routes, Route, Navigate } from 'react-router-dom'
import CasesPage from './pages/CasesPage'
import ProcessingPage from './pages/ProcessingPage'
import ReviewPage from './pages/ReviewPage'
import CompletePage from './pages/CompletePage'
import QueryPage from './pages/QueryPage'
import SummaryPage from './pages/SummaryPage'
import LoginPage from './pages/LoginPage'
import SignupPage from './pages/SignupPage'
import VerifyEmailPage from './pages/VerifyEmailPage'
import { ForgotPasswordPage, ResetPasswordPage } from './pages/PasswordPages'
import ProfilePage from './pages/ProfilePage'
import { useEffect } from 'react'

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const isDev = window.location.hostname === 'localhost'
  if (isDev) return <>{children}</>
  const token = localStorage.getItem('token')
  if (!token) return <Navigate to="/login" replace />
  return <>{children}</>
}

export default function App() {
  useEffect(() => {
    const handleUnload = () => {
      localStorage.removeItem('token')
      localStorage.removeItem('user_name')
      localStorage.removeItem('user_role')
      localStorage.removeItem('user_plan')
      localStorage.removeItem('first_login')
    }
    window.addEventListener('beforeunload', handleUnload)
    return () => window.removeEventListener('beforeunload', handleUnload)
  }, [])

  return (
    <Routes>
      {/* Public */}
      <Route path="/login"           element={<LoginPage />} />
      <Route path="/signup"          element={<SignupPage />} />
      <Route path="/verify"          element={<VerifyEmailPage />} />
      <Route path="/forgot-password" element={<ForgotPasswordPage />} />
      <Route path="/reset-password"  element={<ResetPasswordPage />} />

      {/* Protected */}
      <Route path="/"                   element={<ProtectedRoute><CasesPage /></ProtectedRoute>} />
      <Route path="/profile"            element={<ProtectedRoute><ProfilePage /></ProtectedRoute>} />
      <Route path="/processing/:caseId" element={<ProtectedRoute><ProcessingPage /></ProtectedRoute>} />
      <Route path="/review/:caseId"     element={<ProtectedRoute><ReviewPage /></ProtectedRoute>} />
      <Route path="/complete/:caseId"   element={<ProtectedRoute><CompletePage /></ProtectedRoute>} />
      <Route path="/query/:caseId"      element={<ProtectedRoute><QueryPage /></ProtectedRoute>} />
      <Route path="/summary/:caseId"    element={<ProtectedRoute><SummaryPage /></ProtectedRoute>} />
    </Routes>
  )
}