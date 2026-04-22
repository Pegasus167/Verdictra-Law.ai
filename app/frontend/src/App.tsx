import { Routes, Route, Navigate } from 'react-router-dom'
import CasesPage from './pages/CasesPage'
import ProcessingPage from './pages/ProcessingPage'
import ReviewPage from './pages/ReviewPage'
import CompletePage from './pages/CompletePage'
import QueryPage from './pages/QueryPage'
import SummaryPage from './pages/SummaryPage'
import LoginPage from './pages/LoginPage'

// Auth guard — redirects to /login if no token in localStorage
function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const token = localStorage.getItem('token')
  if (!token) return <Navigate to="/login" replace />
  return <>{children}</>
}

export default function App() {
  return (
    <Routes>
      {/* Public */}
      <Route path="/login" element={<LoginPage />} />

      {/* Protected */}
      <Route path="/" element={<ProtectedRoute><CasesPage /></ProtectedRoute>} />
      <Route path="/processing/:caseId" element={<ProtectedRoute><ProcessingPage /></ProtectedRoute>} />
      <Route path="/review/:caseId" element={<ProtectedRoute><ReviewPage /></ProtectedRoute>} />
      <Route path="/complete/:caseId" element={<ProtectedRoute><CompletePage /></ProtectedRoute>} />
      <Route path="/query/:caseId" element={<ProtectedRoute><QueryPage /></ProtectedRoute>} />
      <Route path="/summary/:caseId" element={<ProtectedRoute><SummaryPage /></ProtectedRoute>} />
    </Routes>
  )
}