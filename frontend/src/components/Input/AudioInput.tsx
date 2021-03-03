import React, { Fragment, useState } from "react";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
  
function AudioInput() {
  // @ts-ignore
  const SpeechRecognition = window.webkitSpeechrecognition;
  // @ts-ignore
	const recognition = new window.webkitSpeechRecognition();
	recognition.continuous = true;
	recognition.interimResult = false;
	recognition.addEventListener("result", (e: { results: { transcript: any; }[][]; }) => {
		console.log(e)
		fetch('http://localhost:8000/input/text/', {
			body: JSON.stringify(e.results[0][0].transcript),
			cache: 'no-cache',
			credentials: 'same-origin',
			headers: {
				'content-type': 'application/json'
			},
			method: 'POST',
			mode: 'cors',
			redirect: 'follow',
			referrer: 'no-referrer',
		})
			.then(async (response) => {
				const res = await response.json();
				console.log(res.lang)
				if (String(res.lang) === "English") {
					// window.$name = 'success' //global variable
          showSuccessToast()
				}
				else {
					// window.$name = 'failure' //global variable
          showFailureToast()
				}
			});
	});
  const showSuccessToast = () => {
    toast.info("Proper Sentence");
  };
  const showFailureToast = () => {
    toast.error("Improper Sentence");
  };
  const onData = (recordedBlob: any) => {
    console.log("chunk of real-time data is: ", recordedBlob);
   
  };

  const onStop = (recordedBlob: any) => {
    console.log("recordedBlob is: ", recordedBlob);
    // const file = new File([recordedBlob],"audio.wav")
    // const blob = new Blob(chunks, { type: 'audio/wav' });

    const formData = new FormData();
    formData.append("File", recordedBlob || "");

    fetch("http://localhost:8000/input/speech/", {
      method: "POST",
      body: formData,
    })
      .then(async (response) => {
        const res = await response.json();
        console.log(res.lang);
        if (String(res.lang) === "English") {
          console.log("adfsadfs");
        } else {
          console.log("adfs");
        }
      })
      .then((result) => {
        console.log("Success:", result);
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  };
  const startRecording = () => {
    recognition.start();
  };

  // const stopRecording = () => {
  //   recognition.stop();
  // };

  return (
    <Fragment>
      <div className="container p-3">
        <div className="row align-items-center">
          <div className="col-md-6 col-6 mx-auto order-md-2">
            Audio Input:
            <button
              onClick={startRecording}
              className="btn btn-primary mx-3"
              type="button"
              id="start"
            >
              Start
            </button>
           
          </div>
        </div>
      </div>
      <ToastContainer />
    </Fragment>
  );
}

export default AudioInput;