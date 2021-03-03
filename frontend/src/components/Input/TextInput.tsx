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

function TextInput() {
  const [text,setText] = useState('');

  function handleTextChange(e: { target: { value: React.SetStateAction<string>; }; }){
    setText(e.target.value)
  }
  const showSuccessToast = () => {
    toast.info("Proper Sentence")
  };
  const showFailureToast = () => {
    toast.error("Improper Sentence")
  };

  function handleSubmit(){  
    console.log(text);
    fetch('http://localhost:8000/input/text/', {
      body: JSON.stringify(text),
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
      });
  }

  return (
    <Fragment>
    <div style={{padding:"50px"}}>
      <InputGroup className="mb-3">
        <FormControl
          placeholder="Code Text"
          aria-label="Code Text"
          aria-describedby="basic-addon2"
          onChange = {handleTextChange}
        />

        <InputGroup.Append>
          <Button variant="success" onClick={handleSubmit} >Submit</Button>
        </InputGroup.Append>
      </InputGroup>
      <ToastContainer />
    </div>
    
    </Fragment>
  );
}

export default TextInput;
