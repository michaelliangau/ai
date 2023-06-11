/* eslint-disable @next/next/no-img-element */

import Link from "next/link";
import { CodeBlock } from "./code-block";

export default function Hero() {
  return (
    <div className="relative isolate pt-14">
      <div className="py-24 sm:py-32 lg:pb-40">
        <div className="mx-auto max-w-7xl px-6 lg:px-8">
          <div className="mx-auto max-w-7xl text-center">
            <h1 className="relative text-4xl font-bold tracking-tighter text-white sm:text-6xl">
              The internet's framework for
              <br />
              <span className="text-green-500">Context Aware AI</span>
            </h1>
            <p className="mt-6 text-lg leading-8 text-neutral-500">
              Connect vector databases to your ChatGPT apps for simple zero-shot document retrieval.
            </p>
            <div className="mt-10 flex items-center justify-center gap-x-6">
              {/* <Link
                href="#"
                className="rounded-md bg-white/10 px-3.5 py-2.5 text-sm font-medium text-white shadow-sm hover:bg-white/20"
              >
                Get started
              </Link> */}
              <Link
                href="https://forms.gle/2hLQTY8ixrCeWFNa9"
                className="text-sm font-medium leading-6 text-white"
              >
                Sign up <span aria-hidden="true">→</span>
              </Link>
            </div>
          </div>
        </div>
      </div>
      <div className="-mt-12 sm:-mt-20 pb-14">
        <CodeBlock />
      </div>
    </div>
  );
}
