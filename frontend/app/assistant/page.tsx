"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function AssistantPage() {
  const router = useRouter();

  useEffect(() => {
    router.replace("/dashboard?tab=assistant");
  }, [router]);

  return (
    <div className="min-h-screen w-full bg-[#fbfcfa] flex items-center justify-center text-slate-500 text-sm">
      Loading Ask Assistant...
    </div>
  );
}
