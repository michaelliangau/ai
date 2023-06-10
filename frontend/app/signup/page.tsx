import {
  MdOutlineRocketLaunch,
  MdOutlineViewInAr,
  MdOutline3P,
} from "react-icons/md";

export default function Signup() {
  return (
    <div className="block box-border flex-grow">
      <form
        action=""
        className="flex md:flex-col items-center justify-center h-full p-6"
      >
        <div className="flex items-center box-border">
          <div
            id="left-side info"
            className="hidden md:flex md:flex-col items-start mr-32"
          >
            <div className="flex flex-col items-start mb-10 text-white">
              <span className="mb-4 uppercase font-mono tracking-wide">
                Lorem Ipsum
              </span>
              <div className="flex flex-col items-start">
                <div className="flex items-center p-2 mb-2">
                  <MdOutlineRocketLaunch className="h-5 w-5 mr-4" />
                  <span className="font-light text-sm">
                    Lorem ipsum dolor sit amet.
                  </span>
                </div>
                <div className="flex items-center p-2 mb-2">
                  <MdOutlineViewInAr className="h-5 w-5 mr-4" />
                  <span className="font-light text-sm">
                    Duis vitae purus faucibus
                  </span>
                </div>
                <div className="flex items-center p-2 mb-2">
                  <MdOutline3P className="h-5 w-5 mr-4" />
                  <span className="font-light text-sm">
                    consectetur adipiscing elit
                  </span>
                </div>
              </div>
            </div>
            <div className="flex flex-col items-start text-white">
              <span className="mb-4 uppercase font-mono tracking-wide">
                Lorem Ipsum
              </span>
              <div className="flex flex-col items-start">
                <div className="flex items-center p-2 mb-2">
                  <MdOutlineRocketLaunch className="h-5 w-5 mr-4" />
                  <span className="font-light text-sm">
                    Lorem ipsum dolor sit amet.
                  </span>
                </div>
                <div className="flex items-center p-2 mb-2">
                  <MdOutlineViewInAr className="h-5 w-5 mr-4" />
                  <span className="font-light text-sm">
                    Duis vitae purus faucibus
                  </span>
                </div>
              </div>
            </div>
          </div>
          <section id="right-side form" className="flex flex-col max-w-md mr-0">
            <div className="block box-border align-baseline mb-3 p-10 bg-neutral-900 rounded-lg shadow">
              <header className="flex flex-col mb-10">
                <h1 className="text-2xl text-white mb-3">Sign-up Form</h1>
                <span className="text-neutral-400 font-light">
                  Nullam aliquet elit eget nisl posuere, ut sollicitudin velit
                  efficitur.
                </span>
              </header>
              <div className="flex flex-col">
                <div className="flex flex-col w-full mb-6">
                  <label
                    htmlFor="email"
                    className="block text-sm leading-6 text-neutral-400"
                  >
                    Email
                  </label>
                  <div className="mt-2">
                    <input
                      type="email"
                      name="email"
                      id="email"
                      className="block w-full rounded-md border-0 py-2.5 text-white bg-neutral-800 focus:ring-2 focus:ring-inset focus:ring-neutral-500 sm:text-sm sm:leading-6"
                    />
                  </div>
                </div>
                <div className="flex flex-col w-full mb-6">
                  <label
                    htmlFor="password"
                    className="block text-sm leading-6 text-neutral-400"
                  >
                    Password
                  </label>
                  <div className="mt-2">
                    <input
                      type="password"
                      name="password"
                      id="password"
                      className="block w-full rounded-md border-0 py-2.5 text-white bg-neutral-800 focus:ring-2 focus:ring-inset focus:ring-neutral-500 sm:text-sm sm:leading-6"
                    />
                  </div>
                </div>
                <div className="flex flex-col w-full mb-6">
                  <label
                    htmlFor="example-1"
                    className="block text-sm font-medium leading-6 text-neutral-400"
                  >
                    Example question?
                  </label>
                  <select
                    id="example-1"
                    name="example-1"
                    className="mt-2 block w-full rounded-md border-0 py-2.5 pl-3 pr-10 text-white bg-neutral-800 focus:ring-2 focus:ring-neutral-500 sm:text-sm sm:leading-6 placeholder:text-neutral-400"
                    defaultValue={""}
                    placeholder="Select an option"
                  >
                    <option value={""} disabled>
                      Select an option
                    </option>
                    <option>Option 1</option>
                    <option>Option 2</option>
                    <option>Option 3</option>
                  </select>
                </div>
                {/* <div className="flex flex-col w-full mb-6">
                  <label
                    htmlFor="example-1"
                    className="block text-sm font-medium leading-6 text-neutral-400"
                  >
                    Question 2
                  </label>
                  <select
                    id="example-2"
                    name="example-2"
                    className="mt-2 block w-full rounded-md border-0 py-2.5 pl-3 pr-10 text-white bg-neutral-800 focus:ring-2 focus:ring-neutral-500 sm:text-sm sm:leading-6 placeholder:text-neutral-400"
                    defaultValue={""}
                  >
                    <option value={""} disabled>
                      Select an option
                    </option>
                    <option>Option 1</option>
                    <option>Option 2</option>
                    <option>Option 3</option>
                  </select>
                </div> */}
                <button
                  type="submit"
                  className="flex w-full justify-center mb-6 rounded-md bg-blue-600 px-3 py-2.5 text-sm leading-6 text-white shadow-sm hover:bg-blue-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600"
                >
                  Continue
                </button>
                <fieldset>
                  <legend className="sr-only">Terms of service</legend>
                  <div className="space-y-5">
                    <div className="relative flex items-start">
                      <div className="flex h-6 items-center">
                        <input
                          id="comments"
                          aria-describedby="comments-description"
                          name="comments"
                          type="checkbox"
                          className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-600"
                        />
                      </div>
                      <div className="ml-3 text-sm leading-6">
                        <span
                          id="comments-description"
                          className="text-neutral-400"
                        >
                          I agree to the Terms of Service
                        </span>
                      </div>
                    </div>
                    <div className="relative flex items-start">
                      <div className="flex h-6 items-center">
                        <input
                          id="candidates"
                          aria-describedby="candidates-description"
                          name="candidates"
                          type="checkbox"
                          className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-600"
                        />
                      </div>
                      <div className="ml-3 text-sm leading-6">
                        <span
                          id="candidates-description"
                          className="text-neutral-400"
                        >
                          (Optional) I would like to receive updates from
                          Silicron
                        </span>
                      </div>
                    </div>
                  </div>
                </fieldset>
              </div>
            </div>
          </section>
        </div>
      </form>
    </div>
  );
}
