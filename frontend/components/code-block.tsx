"use client";

import { ClipboardIcon, CommandLineIcon } from "@heroicons/react/24/outline";

export function CodeBlock() {
  return (
    <div className="w-full mx-auto min-w-0 max-w-3xl mt-4">
      <div className="relative border border-neutral-700 m-4 rounded-lg overflow-hidden shadow-[0_0_50px_rgba(255,255,255,0.0)]">
        <div className="flex items-center border-b border-neutral-700 bg-neutral-950/20 py-2 pl-4 pr-3">
          <div className="flex items-center gap-2 min-w-0 my-0 ml-0 mr-auto">
            <CommandLineIcon className="h-5 w-5 text-white" />
            <span className="inline-block overflow-hidden overflow-ellipsis whitespace-nowrap max-w-full min-w-0 text-sm text-white">
              Terminal
            </span>
          </div>
          <div className="flex gap-1">
            <button className="flex h-8 w-8 items-center justify-center bg-inherit transition-colors">
              <ClipboardIcon className="h-5 w-5 text-white" />
            </button>
          </div>
        </div>
        <pre className="py-5 m-0 overflow-x-auto rounded-b-lg bg-neutral-900/20">
          <code className="text-sm text-white">
            <div className="px-5 leading-6 text-neutral-300">
              <span className="">npx </span>
              <span className="text-green-500">create-next-app@latest</span>
              <br />
              <br />
              <span className="text-orange-300">const </span>
              <span className="">greeting = </span>
              <span className="text-lime-300">"Hello world!"</span>
              <br />
              <span className="">console.</span>
              <span className="text-orange-300">log</span>
              <span className=" text-fuchsia-400">{"("}</span>
              greeting
              <span className=" text-fuchsia-400">{")"}</span>
              <br />
              <br />
              <span className="text-neutral-500">// Output: Hello world!</span>
              <br />
              <br />
              <span className="text-neutral-500">
                This is an example code block.
              </span>
            </div>
          </code>
        </pre>
      </div>
    </div>
  );
}
