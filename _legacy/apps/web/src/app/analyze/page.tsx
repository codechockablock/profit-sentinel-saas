import AnalysisDashboard from "@/components/diagnostic/AnalysisDashboard";

export const metadata = {
  title: "Analyze | Profit Sentinel",
  description: "Full profit leak analysis - detect 11 types of inventory issues",
};

export default function AnalyzePage() {
  return <AnalysisDashboard />;
}
