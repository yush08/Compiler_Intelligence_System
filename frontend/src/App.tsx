import { Toaster } from "sonner";
import { Dashboard } from "./pages/Dashboard";

export default function App() {
  return (
    <>
      <Dashboard />
      <Toaster
        theme="dark"
        position="top-right"
        toastOptions={{
          style: {
            background: "#0b1220",
            border: "1px solid rgba(255,255,255,0.08)",
            color: "#e2e8f0",
          },
        }}
      />
    </>
  );
}
