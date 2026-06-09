import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import JobList from "./pages/JobList";
import NewJob from "./pages/NewJob";
import JobDetail from "./pages/JobDetail";

const queryClient = new QueryClient();

function Layout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-5xl mx-auto px-4 h-14 flex items-center">
          <Link to="/" className="font-bold text-lg text-gray-900 hover:text-indigo-600 transition">
            OneComp
          </Link>
          <span className="ml-2 text-xs text-gray-400 hidden sm:inline">
            LLM Quantization Service
          </span>
        </div>
      </header>
      <main className="max-w-5xl mx-auto px-4 py-8">{children}</main>
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<JobList />} />
            <Route path="/new" element={<NewJob />} />
            <Route path="/jobs/:id" element={<JobDetail />} />
          </Routes>
        </Layout>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
