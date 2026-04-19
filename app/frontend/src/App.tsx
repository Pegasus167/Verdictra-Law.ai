import { Routes, Route } from 'react-router-dom'
import CasesPage from './pages/CasesPage'
import ProcessingPage from './pages/ProcessingPage'
import ReviewPage from './pages/ReviewPage'
import CompletePage from './pages/CompletePage'
import QueryPage from './pages/QueryPage'
import SummaryPage from './pages/SummaryPage'


export default function App() {
  return (
    <Routes>
      <Route path="/" element={<CasesPage />} />
      <Route path="/processing/:caseId" element={<ProcessingPage />} />
      <Route path="/review/:caseId" element={<ReviewPage />} />
      <Route path="/complete/:caseId" element={<CompletePage />} />
      <Route path="/query/:caseId" element={<QueryPage />} />
      <Route path="/summary/:caseId" element={<SummaryPage />} />
    </Routes>
  )
}