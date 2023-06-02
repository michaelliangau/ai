/* eslint-disable @next/next/no-img-element */

import Link from "next/link";
import { CodeBlock } from "./code-block";

export default function Hero() {
  return (
    <div className="relative isolate pt-14">
      {/* <div
        className="absolute inset-x-0 -top-40 -z-10 transform-gpu overflow-hidden blur-3xl sm:-top-80"
        aria-hidden="true"
      >
        <div
          className="relative left-[calc(50%-11rem)] aspect-[1155/678] w-[36.125rem] -translate-x-1/2 rotate-[30deg] bg-gradient-to-tr from-[#ff80b5] to-[#9089fc] opacity-20 sm:left-[calc(50%-30rem)] sm:w-[72.1875rem]"
          style={{
            clipPath:
              "polygon(74.1% 44.1%, 100% 61.6%, 97.5% 26.9%, 85.5% 0.1%, 80.7% 2%, 72.5% 32.5%, 60.2% 62.4%, 52.4% 68.1%, 47.5% 58.3%, 45.2% 34.5%, 27.5% 76.7%, 0.1% 64.9%, 17.9% 100%, 27.6% 76.8%, 76.1% 97.7%, 74.1% 44.1%)",
          }}
        />
      </div> */}
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
      {/* <div
        className="absolute inset-x-0 top-[calc(100%-14rem)] -z-10 transform-gpu overflow-hidden blur-3xl sm:top-[calc(100%-36rem)]"
        aria-hidden="true"
      >
        <div
          className="relative left-[calc(50%+3rem)] aspect-[1155/678] w-[36.125rem] -translate-x-1/2 bg-gradient-to-tr from-[#ff80b5] to-[#9089fc] opacity-20 sm:left-[calc(50%+36rem)] sm:w-[72.1875rem]"
          style={{
            clipPath:
              "polygon(74.1% 44.1%, 100% 61.6%, 97.5% 26.9%, 85.5% 0.1%, 80.7% 2%, 72.5% 32.5%, 60.2% 62.4%, 52.4% 68.1%, 47.5% 58.3%, 45.2% 34.5%, 27.5% 76.7%, 0.1% 64.9%, 17.9% 100%, 27.6% 76.8%, 76.1% 97.7%, 74.1% 44.1%)",
          }}
        />
      </div> */}
    </div>
  );
}
