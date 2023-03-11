"use client";
import { useState } from "react";
import Image from "next/image";

const CONFIG = {
  backend_url:
    "https://4do5ytwgq1.execute-api.ap-south-1.amazonaws.com/dev/inference",
};

export default function Home() {
  const [imageBase64, setImageBase64] = useState("");
  const [results, setResults] = useState("");
  const [inferencing, setInferencing] = useState(false);

  const runInference = async () => {
    if (!!!imageBase64) return;

    const _infer = async () => {
      var myHeaders = new Headers();
      myHeaders.append("Content-Type", "application/json");

      var raw = JSON.stringify({
        body: imageBase64,
      });

      var requestOptions = {
        method: "POST",
        headers: myHeaders,
        body: raw,
      };

      const response = await fetch(CONFIG.backend_url, requestOptions);
      const result = await response.json();
      setResults(JSON.stringify(result[0], null, "\t"));
    };

    try {
      setInferencing(true);
      await _infer();
    } catch (e) {
      console.error("error :: ", e);
    } finally {
      setInferencing(false);
    }
  };

  const LoadingSvg = () => (
    <svg
      aria-hidden="true"
      role="status"
      className="inline mr-3 w-4 h-4 text-white animate-spin"
      viewBox="0 0 100 101"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M100 50.5908C100 78.2051 77.6142 100.591 50 100.591C22.3858 100.591 0 78.2051 0 50.5908C0 22.9766 22.3858 0.59082 50 0.59082C77.6142 0.59082 100 22.9766 100 50.5908ZM9.08144 50.5908C9.08144 73.1895 27.4013 91.5094 50 91.5094C72.5987 91.5094 90.9186 73.1895 90.9186 50.5908C90.9186 27.9921 72.5987 9.67226 50 9.67226C27.4013 9.67226 9.08144 27.9921 9.08144 50.5908Z"
        fill="#E5E7EB"
      />
      <path
        d="M93.9676 39.0409C96.393 38.4038 97.8624 35.9116 97.0079 33.5539C95.2932 28.8227 92.871 24.3692 89.8167 20.348C85.8452 15.1192 80.8826 10.7238 75.2124 7.41289C69.5422 4.10194 63.2754 1.94025 56.7698 1.05124C51.7666 0.367541 46.6976 0.446843 41.7345 1.27873C39.2613 1.69328 37.813 4.19778 38.4501 6.62326C39.0873 9.04874 41.5694 10.4717 44.0505 10.1071C47.8511 9.54855 51.7191 9.52689 55.5402 10.0491C60.8642 10.7766 65.9928 12.5457 70.6331 15.2552C75.2735 17.9648 79.3347 21.5619 82.5849 25.841C84.9175 28.9121 86.7997 32.2913 88.1811 35.8758C89.083 38.2158 91.5421 39.6781 93.9676 39.0409Z"
        fill="currentColor"
      />
    </svg>
  );

  return (
    <div className="flex min-h-screen flex-col items-center justify-center py-2">
      <main className="flex w-full flex-1 flex-col items-center justify-center px-20 text-center">
        <h1 className="text-6xl font-bold">
          The Snap-ination Station{" "}
          <a className="text-blue-600" href="https://nextjs.org">
            Serverless
          </a>
        </h1>
        <div className="mt-10 text-lg">
          Select a PNG Image from{" "}
          <a
            href="https://github.com/vivekchaudhary07/S9-Serverless-Frontend/blob/main/input.jpg"
            target="_blank"
            className="text-blue-600"
          >
            Example
          </a>
        </div>
        <div className="mt-10 flex flex-row space-x-5 text-xl">
          <div>
            <input
              className="block w-full text-sm text-gray-900 bg-gray-50 rounded-lg border border-gray-300 cursor-pointer dark:text-gray-400 focus:outline-none dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400"
              aria-describedby="file_input_help"
              id="file_input"
              type="file"
              accept="image/png"
              onChange={(e) => {
                console.log("got files :: ", e.target.files);

                if (!e.target.files) return;
                if (e.target.files?.length == 0) return;

                const imageFile = e.target.files[0];

                var reader = new FileReader();
                var baseString;
                reader.onloadend = function () {
                  baseString = reader.result;
                  if (baseString) setImageBase64(baseString.toString());
                };
                reader.readAsDataURL(imageFile);

                setResults("");
              }}
            />
          </div>
          <div>
            <button
              type="button"
              className="text-white bg-gray-800 hover:bg-gray-900 focus:outline-none focus:ring-4 focus:ring-gray-300 font-medium rounded-lg text-sm px-5 py-2.5 mr-2 mb-2 dark:bg-gray-800 dark:hover:bg-gray-700 dark:focus:ring-gray-700 dark:border-gray-700"
              onClick={runInference}
              disabled={!!!imageBase64 || inferencing}
            >
              {inferencing ? (
                <>
                  <LoadingSvg />
                  Running
                </>
              ) : (
                "Run 🚀"
              )}
            </button>
          </div>
        </div>
        {imageBase64 && (
          <div className="mt-10">
            <img src={imageBase64} alt="Input" width={128} height={128} />
          </div>
        )}
        {results && (
          <div className="mt-10 ">
            <code>{results}</code>
          </div>
        )}
      </main>

      <footer className="flex h-24 w-full items-center justify-center border-t">
        <a
          className="flex items-center justify-center gap-2 text-xl"
          href="https://theschoolofai.in/"
          target="_blank"
          rel="noopener noreferrer"
        >
          EMLO 2.0 🤖 The School of AI
        </a>
      </footer>
    </div>
  );
}
