import DiagnosticDashboard from "@/components/diagnostic/DiagnosticDashboard";

export const metadata = {
  title: "Diagnostic | Profit Sentinel",
  description: "Interactive shrinkage diagnostic - identify process issues vs real losses",
};

export default function DiagnosticPage() {
  return <DiagnosticDashboard />;
}
