import React, { Fragment, useState } from "react";
import logo from "./logo.svg";
import {
  Button,
  Form,
  FormControl,
  InputGroup,
  Nav,
  Navbar,
  NavDropdown,
} from "react-bootstrap";

import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
function FileInput() {
	const [selectedFile, setSelectedFile] = useState();

  const showSuccessToast = () => {
    toast.info("Proper Sentence")
  };
  const showFailureToast = () => {
    toast.error("Improper Sentence")
  };
  const changeHandler = (event:any) => {
		setSelectedFile(event.target.files[0]);
	};
  
  const handleSubmission = () => {
		const formData = new FormData();

    formData.append('File', selectedFile || '');
		fetch(
			'http://localhost:8000/input/file/',
			{
				method: 'POST',
				body: formData,
			}
		)
			.then(async(response)=>{
        const res = await response.json();
        console.log(res.lang)
        if(String(res.lang) === "English")
        {
          console.log("adfsadfs")
          showSuccessToast();
        }
        else{
          showFailureToast();
        }
      })
			.then((result) => {
				console.log('Success:', result);
			})
			.catch((error) => {
				console.error('Error:', error);
			});
	};
  return (
    <Fragment>
        <div style={{padding:"50px"}}>
            <div className="col-md-6 col-6 mx-auto order-md-2">
                File Input: <input type="file" name="fileInput" id="fileInput" accept=".txt" onChange={changeHandler}/>
                <Button type="file" variant="success" onClick={handleSubmission}>Submit</Button>
            </div>
        </div>
        <ToastContainer/>
    </Fragment>
  );
};
export default FileInput;
