import { Fragment, useState } from "react";
import {
    Button,
} from "react-bootstrap";

import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import CodeEditor from "../../shared/CodeEditor/CodeEditor";
function FileInput() {
    const [selectedFile, setSelectedFile] = useState();
    const [disabled, setDisabled] = useState(true);
    const [codeText, setCodeText] = useState(null);
    const showSuccessToast = () => {
        toast.info("Proper Sentence")
    };
    const showFailureToast = () => {
        toast.error("Improper Sentence")
    };
    const changeHandler = (event: any) => {
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
            .then(async (response) => {
                const res = await response.json();
                console.log(res.lang)
                if (String(res.lang) === "English") {
                    if (localStorage.getItem("code") === null) {
                        localStorage.setItem("code", res.output);
                    }
                    //@ts-ignore
                    else if (localStorage.getItem("code").charAt(localStorage.getItem("code").length-1) === ":"){
                        localStorage.setItem(
                            "code",
                            localStorage.getItem("code") + "\n" + "\t" + res.output
                        );
                    }
                    else{
                        localStorage.setItem(
                            "code",
                            localStorage.getItem("code") + "\n" +res.output
                        );
                    }
                    //@ts-ignore
                    setCodeText(localStorage.getItem("code"));
                    setDisabled(false)
                    showSuccessToast();
                }
                else {
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
            <div style={{ textAlign: "center", paddingTop: "50px" }}>
                <div className="col-md-6 col-12 mx-auto order-md-2">
                    <input type="file" style={{ margin: "20px", paddingLeft: "30px" }} name="fileInput" id="fileInput" accept=".txt" onChange={changeHandler} />
                    <Button type="file" variant="success" onClick={handleSubmission}>Submit</Button>
                </div>
            </div>
            <CodeEditor disabledVal={disabled} codeText={codeText} />
            <ToastContainer />
        </Fragment>
    );
};
export default FileInput;
