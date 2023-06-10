import Header from "@/components/header";

export const metadata = {
  title: "Silicron",
};

interface MarketingLayoutProps {
  children: React.ReactNode;
}

export default function MarketingLayout({ children }: MarketingLayoutProps) {
  return (
    <div className="bg-neutral-950">
      <Header />
      {children}
    </div>
  );
}
