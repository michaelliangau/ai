import Header from "@/components/header";
import { CommandLineIcon } from "@heroicons/react/24/outline";
import Link from "next/link";

export const metadata = {
  title: "Silicron",
};

interface SignupLayoutProps {
  children: React.ReactNode;
}

export default function SignupLayout({ children }: SignupLayoutProps) {
  return (
    <div className="bg-neutral-950">
      <div className="flex sticky justify-between align-baseline box-border h-12 z-10 shrink-0 top-0 bg-neutral-900">
        <Link href={"/"} className="flex pl-8 lg:flex-1 items-center">
          {/* <div className="flex pl-8 lg:flex-1 items-center"> */}
          <CommandLineIcon className="h-6 w-6 text-white mr-2 stroke-1.5" />
          <h1 className="text-white text-lg tracking-wide">Silicron</h1>
          {/* </div> */}
        </Link>
      </div>
      {children}
    </div>
  );
}
