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
              Simplifying Large Context Language Models
            </h1>
            <p className="mt-6 text-lg leading-8 text-neutral-500">
              Silicron is a framework designed to extend the capabilities of
              OpenAI's GPT-3 model, making it effortless for developers to
              integrate additional data into their applications, without having
              to handle the intricacies of context length and context retrieval.
            </p>
            <div className="mt-10 flex items-center justify-center gap-x-6">
              <Link
                href="#"
                className="rounded-md bg-white/10 px-3.5 py-2.5 text-sm font-medium text-white shadow-sm hover:bg-white/20"
              >
                Get started
              </Link>
              <Link
                href="/signup"
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
